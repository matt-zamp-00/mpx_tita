"""Headless benchmark of the TITA MPC + WBC baseline (Phase 1 of the residual-RL workflow).

Runs a fixed set of scripted command scenarios on each MuJoCo scene with the exact control loop
of mpx/examples/mjx_tita.py (MPC at config.mpc_frequency, WBC at config.whole_body_frequency,
physics at config.simulation_frequency) and writes one metrics row per (scene, scenario).

The same scenarios and metrics are meant to be reused later to evaluate MPC + residual policy,
so keep them fixed once the baseline numbers are in the thesis (add new ones instead of editing).

Usage (from the mpx_tita root):
    pixi run python experiments/baseline_eval.py                       # all scenes, all scenarios
    pixi run python experiments/baseline_eval.py --scenes flat --scenarios stand vx_step
    pixi run python experiments/baseline_eval.py --save-traj           # also store per-step logs
    pixi run python experiments/baseline_eval.py --repeats 5           # success rate, mean ± std

Output: experiments/results/<timestamp>_<tag>/
    meta.json     git commit, settings, scenarios
    metrics.csv   one row per run (scene, scenario, repeat)
    summary.csv   per (scene, scenario): success rate, fall time and metrics mean/std over repeats
    traj/*.npz    per-step logs at the WBC rate (--save-traj)
"""
import argparse
import collections
import csv
import datetime
import json
import os
import subprocess
import sys
import time

dir_path = os.path.dirname(os.path.realpath(__file__))
repo_root = os.path.abspath(os.path.join(dir_path, ".."))
sys.path.append(repo_root)
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

import mpx.config.config_tita as config
import mpx.utils.mpc_wrapper_dfcip as mpc_wrapper_dfcip
import mpx.utils.sim as sim_utils
from mpx.examples.mjx_tita import _gather_raw_state, _mpc_due, _reset_to_initial_state

jax.config.update("jax_enable_x64", True)

H0 = float(config.com_z_to_track)

# ---------------------------------------------------------------------------------------------
# Scenarios: piecewise-constant commands. Each segment is (t_start [s], vx [m/s], wz [rad/s],
# height [m]). vz (2nd MPC input) is always 0. The last segment holds until `duration`.
# ---------------------------------------------------------------------------------------------
SCENARIOS = {
    "stand":     dict(duration=6.0,  segments=[(0.0, 0.0, 0.0, H0)]),
    "vx_step":   dict(duration=10.0, segments=[(0.0, 0.0, 0.0, H0), (1.0, 1.0, 0.0, H0),
                                               (6.0, 0.0, 0.0, H0)]),
    "vx_fast":   dict(duration=10.0, segments=[(0.0, 0.0, 0.0, H0), (1.0, 2.0, 0.0, H0),
                                               (6.0, 0.0, 0.0, H0)]),
    "backward":  dict(duration=9.0,  segments=[(0.0, 0.0, 0.0, H0), (1.0, -0.5, 0.0, H0),
                                               (5.0, 0.0, 0.0, H0)]),
    "turn":      dict(duration=9.0,  segments=[(0.0, 0.0, 0.0, H0), (1.0, 0.0, 1.0, H0),
                                               (5.0, 0.0, 0.0, H0)]),
    "arc":       dict(duration=10.0, segments=[(0.0, 0.0, 0.0, H0), (1.0, 0.8, 0.5, H0),
                                               (6.0, 0.0, 0.0, H0)]),
    "height":    dict(duration=10.0, segments=[(0.0, 0.0, 0.0, H0), (1.0, 0.0, 0.0, 0.30),
                                               (5.0, 0.0, 0.0, 0.38)]),
}
SCENES = ["flat", "rough", "perlin", "stairs"]

SETTLE_S = 1.5          # seconds after each command change excluded from the "steady" errors
FALL_TILT_RAD = 1.0     # base tilt (angle between base z-axis and world z) that counts as a fall
FALL_COM_HEIGHT = 0.15  # CoM height above the wheel-contact midpoint that counts as a fall


def command_at(segments, t):
    seg = segments[0]
    for s in segments:
        if t >= s[0]:
            seg = s
    _, vx, wz, h = seg
    return np.array([vx, 0.0, wz, h], dtype=np.float64)


def time_since_change(segments, t):
    starts = [s[0] for s in segments if s[0] <= t]
    return t - max(starts)


def git_info(path):
    def run(*args):
        try:
            return subprocess.check_output(["git", "-C", path, *args], text=True).strip()
        except Exception:
            return None
    return dict(commit=run("rev-parse", "HEAD"), branch=run("rev-parse", "--abbrev-ref", "HEAD"),
                dirty=bool(run("status", "--porcelain")))


def apply_overrides(cfg, items):
    """Apply --set overrides to the config module before the controller is built.
    "key=value" sets cfg.key; "W_mpc.key=value" / "W_wbc.key=value" set a weight in every phase.
    Values are parsed as Python literals (1e6, 24, [0, 0, 1e2]), else kept as strings (dcm)."""
    import ast
    applied = {}
    for item in items or []:
        key, _, raw = item.partition("=")
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        if key.startswith(("W_mpc.", "W_wbc.")):
            table, name = key.split(".", 1)
            for phase in getattr(cfg, table).values():
                if name not in phase["weights"]:
                    raise KeyError(f"--set: {table} has no weight {name}")
                phase["weights"][name] = value
        else:
            if not hasattr(cfg, key):
                raise KeyError(f"--set: config has no attribute {key}")
            setattr(cfg, key, value)
        applied[key] = value
    return applied


def rmse(x):
    x = np.asarray(x)
    return float(np.sqrt(np.mean(x ** 2))) if x.size else float("nan")


def run_episode(mpc, model, scenario, pert=None, residual_fn=None):
    """One episode; returns (metrics dict, trajectory dict). pert: perturbations.Perturbation of
    the simulated robot (None = nominal); the controller always keeps its nominal model.
    residual_fn(t, x0, data, mpc_state) -> task-space residual (8,) or None, called at every WBC
    step (see mpc_wrapper_dfcip.RESIDUAL_LAYOUT); None = the baseline."""
    if pert is not None:
        model = pert.apply_model(model)
    data = mujoco.MjData(model)
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config.base_body_name)
    contact_ids = sim_utils.geom_ids(model, config.contact_frame)
    act_dof = model.jnt_dofadr[model.actuator_trnid[:, 0]]
    sim_frequency = float(config.simulation_frequency)
    dt = model.opt.timestep
    wbc_period = config.wbc_period_steps
    n_steps = int(round(scenario["duration"] / dt))
    segments = scenario["segments"]

    _reset_to_initial_state(model, data)
    if pert is not None:
        pert.apply_initial_state(model, data)
    latency = pert.latency_steps(dt) if pert is not None else 0
    pending = collections.deque()  # (physics step at which to apply, torques)
    mpc_state = mpc.reset()
    theta_prev = 0.0
    tau = jnp.zeros((1, model.nu))
    x0 = None
    tita_state = None

    log = {k: [] for k in ("t", "cmd", "v", "w", "com_h", "tilt", "tau", "power")}
    mpc_ms, wbc_ms = [], []
    fell_at = None

    for k in range(n_steps):
        t = k * dt
        cmd = command_at(segments, t)
        mpc_due = _mpc_due(k, sim_frequency, config.mpc_frequency)
        wbc_due = k % wbc_period == 0

        if mpc_due or wbc_due:
            raw = _gather_raw_state(model, data, base_body_id, contact_ids)
            tita_state, x0, theta_prev = mpc.process_state(*raw, theta_prev)

        if mpc_due:
            tic = time.perf_counter()
            mpc_state, _ = mpc.run(mpc_state, x0[None, :], jnp.asarray(cmd)[None, :])
            jax.block_until_ready(mpc_state.sol.a)
            mpc_ms.append(1e3 * (time.perf_counter() - tic))

        if wbc_due:
            tic = time.perf_counter()
            residual = residual_fn(t, x0, data, mpc_state) if residual_fn is not None else None
            mpc_state, tau, _, _, _, _ = mpc.whole_body_run(
                mpc_state, x0,
                jnp.asarray(data.qpos.copy())[None, :], jnp.asarray(data.qvel.copy())[None, :],
                tita_state[6:9][None, :], tita_state[9:12][None, :],
                tita_state[12:15][None, :], tita_state[15:18][None, :],
                residual=None if residual is None else jnp.asarray(residual, dtype=jnp.float64)[None, :],
            )
            tau_np = np.asarray(tau[0])
            wbc_ms.append(1e3 * (time.perf_counter() - tic))
            pending.append((k + latency, tau_np))

            # Log at the WBC rate (100 Hz): cheap and enough for tracking metrics.
            x0_np = np.asarray(x0)
            base_zaxis = data.xmat[base_body_id].reshape(3, 3)[:, 2]
            tilt = float(np.arccos(np.clip(base_zaxis[2], -1.0, 1.0)))
            com_h = float(x0_np[2] - x0_np[8])  # CoM z above wheel-contact midpoint z
            power = float(np.sum(np.abs(data.actuator_force * data.qvel[act_dof])))
            log["t"].append(t); log["cmd"].append(cmd); log["v"].append(x0_np[11])
            log["w"].append(x0_np[12]); log["com_h"].append(com_h); log["tilt"].append(tilt)
            log["tau"].append(tau_np); log["power"].append(power)

            if not np.all(np.isfinite(tau_np)) or tilt > FALL_TILT_RAD or com_h < FALL_COM_HEIGHT:
                fell_at = t
                break

        while pending and pending[0][0] <= k:   # torques whose latency has elapsed
            data.ctrl[:] = pending.popleft()[1]
        if pert is not None:
            data.xfrc_applied[base_body_id] = pert.push_wrench(t, data.xmat[base_body_id])
        mujoco.mj_step(model, data)

    traj = {k: np.asarray(v) for k, v in log.items()}
    t_arr, cmd_arr = traj["t"], traj["cmd"]
    steady = np.array([time_since_change(segments, ti) >= SETTLE_S for ti in t_arr], dtype=bool)
    e_v = traj["v"] - cmd_arr[:, 0]
    e_w = traj["w"] - cmd_arr[:, 2]
    e_h = traj["com_h"] - cmd_arr[:, 3]
    dt_log = 1.0 / config.whole_body_frequency
    # Drop the first solve of each type (it may include tracing/compilation).
    mpc_t, wbc_t = np.asarray(mpc_ms[1:]), np.asarray(wbc_ms[1:])

    metrics = dict(
        fell=fell_at is not None,
        fall_time_s=fell_at,
        sim_time_s=float(t_arr[-1]) if t_arr.size else 0.0,
        rmse_v=rmse(e_v), rmse_v_steady=rmse(e_v[steady]),
        rmse_w=rmse(e_w), rmse_w_steady=rmse(e_w[steady]),
        rmse_h=rmse(e_h), rmse_h_steady=rmse(e_h[steady]),
        max_tilt_deg=float(np.degrees(traj["tilt"].max())) if t_arr.size else float("nan"),
        mean_abs_tau=float(np.mean(np.abs(traj["tau"]))) if t_arr.size else float("nan"),
        peak_abs_tau=float(np.max(np.abs(traj["tau"]))) if t_arr.size else float("nan"),
        energy_J=float(np.sum(traj["power"]) * dt_log),
        mpc_ms_mean=float(mpc_t.mean()) if mpc_t.size else float("nan"),
        mpc_ms_p99=float(np.percentile(mpc_t, 99)) if mpc_t.size else float("nan"),
        wbc_ms_mean=float(wbc_t.mean()) if wbc_t.size else float("nan"),
        wbc_ms_p99=float(np.percentile(wbc_t, 99)) if wbc_t.size else float("nan"),
    )
    return metrics, traj


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", nargs="+", default=SCENES, choices=SCENES)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    parser.add_argument("--tag", default="baseline", help="label appended to the results folder")
    parser.add_argument("--out", default=os.path.join(dir_path, "results"))
    parser.add_argument("--save-traj", action="store_true", help="store per-step logs as .npz")
    parser.add_argument("--repeats", type=int, default=1,
                        help="runs per (scene, scenario); the runs differ only through the "
                             "non-determinism of JAX on the GPU, so this measures how reliable a result is")
    parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE",
                        help="config overrides, e.g. N=24 terminal_stability=dcm W_mpc.w_stab=1e6")
    args = parser.parse_args()
    overrides = apply_overrides(config, args.set)

    os.chdir(repo_root)  # the wrapper's persistent JAX cache is ./jax_cache
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out, f"{stamp}_{args.tag}")
    os.makedirs(out_dir, exist_ok=True)

    meta = dict(
        date=stamp, tag=args.tag, git=git_info(repo_root), scenes=args.scenes, repeats=args.repeats, overrides=overrides,
        scenarios={k: SCENARIOS[k] for k in args.scenarios},
        settle_s=SETTLE_S, fall_tilt_rad=FALL_TILT_RAD, fall_com_height=FALL_COM_HEIGHT,
        config=dict(simulation_frequency=config.simulation_frequency,
                    whole_body_frequency=config.whole_body_frequency,
                    mpc_frequency=config.mpc_frequency, dt_mpc=config.dt_mpc, N=config.N,
                    com_z_to_track=config.com_z_to_track),
        jax_devices=[str(d) for d in jax.devices()], mujoco_version=mujoco.__version__,
    )
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    if meta["git"]["dirty"]:
        print("WARNING: mpx_tita has uncommitted changes; these results are not reproducible "
              "from the recorded commit alone.")

    print("Building and compiling the MPC + WBC (first run can take minutes)...")
    mpc = mpc_wrapper_dfcip.BatchedMPCControllerWrapper(config, n_env=1)

    rows = []
    for scene in args.scenes:
        model = mujoco.MjModel.from_xml_path(os.path.join(repo_root, f"mpx/data/tita/scene_{scene}.xml"))
        model.opt.timestep = config.dt_sim
        for name in args.scenarios:
            for rep in range(args.repeats):
                tic = time.perf_counter()
                metrics, traj = run_episode(mpc, model, SCENARIOS[name])
                row = dict(scene=scene, scenario=name, repeat=rep, **metrics)
                rows.append(row)
                status = f"FELL at {metrics['fall_time_s']:.2f}s" if metrics["fell"] else "ok"
                print(f"[{scene:6s} {name:9s} #{rep}] {status:14s} rmse_v {metrics['rmse_v_steady']:.3f} "
                      f"rmse_w {metrics['rmse_w_steady']:.3f} rmse_h {metrics['rmse_h_steady']:.3f} "
                      f"tilt {metrics['max_tilt_deg']:5.1f}deg  mpc {metrics['mpc_ms_mean']:.1f}ms "
                      f"({time.perf_counter() - tic:.0f}s wall)", flush=True)
                if args.save_traj:
                    os.makedirs(os.path.join(out_dir, "traj"), exist_ok=True)
                    np.savez_compressed(os.path.join(out_dir, "traj", f"{scene}_{name}_{rep}.npz"), **traj)

    write_csv(os.path.join(out_dir, "metrics.csv"), rows)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(rows, f, indent=2)
    summary = summarize(rows)
    write_csv(os.path.join(out_dir, "summary.csv"), summary)
    print_summary(summary)
    print(f"\nResults written to {out_dir}")


# Metrics summarized over repeats. Tracking/effort metrics only over the repeats that did not
# fall (a fallen run is cut short, so its errors are not comparable); fall time over those that fell.
SUMMARY_METRICS = ["rmse_v_steady", "rmse_w_steady", "rmse_h_steady", "max_tilt_deg",
                   "mean_abs_tau", "peak_abs_tau", "energy_J", "mpc_ms_mean", "wbc_ms_mean"]


def summarize(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["scene"], r["scenario"]), []).append(r)
    out = []
    for (scene, scenario), rs in groups.items():
        ok = [r for r in rs if not r["fell"]]
        fell = [r for r in rs if r["fell"]]
        s = dict(scene=scene, scenario=scenario, n=len(rs), n_ok=len(ok),
                 success_rate=len(ok) / len(rs))
        ft = np.array([r["fall_time_s"] for r in fell], dtype=float)
        s["fall_time_mean"] = float(ft.mean()) if ft.size else float("nan")
        s["fall_time_std"] = float(ft.std()) if ft.size else float("nan")
        for m in SUMMARY_METRICS:
            v = np.array([r[m] for r in ok], dtype=float)
            v = v[np.isfinite(v)]
            s[f"{m}_mean"] = float(v.mean()) if v.size else float("nan")
            s[f"{m}_std"] = float(v.std()) if v.size else float("nan")
        out.append(s)
    return out


def print_summary(summary):
    print("\nSummary (tracking errors over the runs that did not fall):")
    print(f"{'scene':6s} {'scenario':9s} {'success':>8s} {'fall t [s]':>12s} {'rmse_v':>14s} "
          f"{'rmse_w':>14s} {'rmse_h':>14s}")
    for s in summary:
        ft = (f"{s['fall_time_mean']:.2f}±{s['fall_time_std']:.2f}"
              if np.isfinite(s["fall_time_mean"]) else "-")
        cells = [f"{s[m + '_mean']:.3f}±{s[m + '_std']:.3f}" if s["n_ok"] else "-"
                 for m in ("rmse_v_steady", "rmse_w_steady", "rmse_h_steady")]
        print(f"{s['scene']:6s} {s['scenario']:9s} {s['n_ok']:>3d}/{s['n']:<4d} {ft:>12s} "
              f"{cells[0]:>14s} {cells[1]:>14s} {cells[2]:>14s}")


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
