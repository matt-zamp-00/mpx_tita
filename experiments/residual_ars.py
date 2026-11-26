"""Phase 3, step 3: a short residual training run that validates the hook and the action space.

Question: can a task-space residual (method B, mpc_wrapper_dfcip.RESIDUAL_LAYOUT) on top of the
*original* controller get the robot over the curbs of benchmark v2, where the baseline falls even
without perturbations? If yes, the hook works end to end and the action space can close the
largest gap of the baseline. If no, the action space (or the WBC) is the bottleneck, before any
large training run.

Method: Augmented Random Search (Mania et al., 2018, ARS V1-t) on a linear policy. It needs
only episode returns, so it runs on the existing single-environment loop (about 2 s of wall time
per simulated second on the laptop). PPO needs 1e7+ steps and waits for Phase 4 (batching).

Policy:  a = scale * clip(W @ obs, -1, 1),   W initialised to 0 (exactly the baseline).
Observations: only signals the real robot has (command, controller state estimate, IMU, joint
velocities), with fixed normalisation; see `observe`.
Return per episode: mean over the *full* duration of 1 - |e_v| - 0.5|e_w| - 0.01 |a/scale|^2
per WBC step (each term clipped to [0, 1]), so a fall loses the remaining steps.

Usage (from the mpx_tita root):
    pixi run python experiments/residual_ars.py train --iters 15          # ~1.5 h on the laptop
    pixi run python experiments/residual_ars.py eval  results/<run>/W_best.npy

Output: experiments/results/<timestamp>_ars_<tag>/{meta.json, log.csv, W_iter*.npy, W_best.npy}
"""
import argparse
import datetime
import json
import os
import time

import mujoco
import numpy as np

import baseline_eval as be
import robustness_eval as re_

SCALE = np.array([2.0, 2.0, 2.0, 5.0, 5.0, 5.0, 3.0, 3.0])   # m/s^2, rad/s^2, m/s^2 per channel
TRAIN_CASES = [("steps", "vx_step"), ("steps", "backward"), ("steps", "arc")]
N_OBS = 18


def make_policy(W, scenario, model, base_id, act_dof, log=None):
    """residual_fn for be.run_episode: a = scale * clip(W @ obs, -1, 1)."""
    segments = scenario["segments"]

    def observe(t, x0, data):
        x0 = np.asarray(x0)
        cmd = be.command_at(segments, t)
        R = data.xmat[base_id].reshape(3, 3)
        g_body = R.T @ np.array([0.0, 0.0, -1.0])           # IMU gravity direction
        omega = data.qvel[3:6]                               # IMU angular velocity (base frame)
        return np.concatenate([
            [(x0[11] - cmd[0]) / 0.5, (x0[12] - cmd[2]) / 0.5, (x0[2] - x0[8] - cmd[3]) / 0.05,
             cmd[0], cmd[2]],
            g_body[:2] / 0.2, omega / 1.0,
            data.qvel[act_dof] / 10.0,                       # 6 leg joints + 2 wheels
        ])

    def residual_fn(t, x0, data, mpc_state):
        u = np.clip(W @ observe(t, x0, data), -1.0, 1.0)
        if log is not None:
            log.append(u)
        return SCALE * u

    return residual_fn


def episode_return(metrics, traj, scenario, actions):
    n_full = int(round(scenario["duration"] * be.config.whole_body_frequency))
    if traj["t"].size == 0:
        return 0.0
    e_v = np.abs(traj["v"] - traj["cmd"][:, 0])
    e_w = np.abs(traj["w"] - traj["cmd"][:, 2])
    a2 = np.sum(np.asarray(actions) ** 2, axis=1) if len(actions) else 0.0
    r = 1.0 - np.minimum(e_v, 1.0) - 0.5 * np.minimum(e_w, 1.0) - 0.01 * np.minimum(a2, 1.0)
    return float(np.sum(r[: n_full]) / n_full)


class Runner:
    def __init__(self, state="nominal"):
        os.chdir(be.repo_root)
        self.mpc = be.mpc_wrapper_dfcip.BatchedMPCControllerWrapper(be.config, n_env=1)
        self.models = {}
        self.state = state

    def model(self, scene):
        if scene not in self.models:
            self.models[scene] = re_.load_model(scene)
        return self.models[scene]

    def rollout(self, W, scene, scenario_name, pert=None):
        model = self.model(scene)
        sc = be.SCENARIOS[scenario_name]
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, be.config.base_body_name)
        act_dof = model.jnt_dofadr[model.actuator_trnid[:, 0]]
        actions = []
        fn = make_policy(W, sc, model, base_id, act_dof, actions)
        m, traj = be.run_episode(self.mpc, model, sc, pert, residual_fn=fn, state=self.state)
        return episode_return(m, traj, sc, actions), m

    def evaluate(self, W, cases):
        rets, ms = [], []
        for scene, scen in cases:
            r, m = self.rollout(W, scene, scen)
            rets.append(r); ms.append(m)
        return float(np.mean(rets)), ms


def train(args):
    rng = np.random.default_rng(args.seed)
    runner = Runner(args.state)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(be.repo_root, "experiments", "results", f"{stamp}_ars_{args.tag}")
    os.makedirs(out)
    meta = dict(vars(args), cases=TRAIN_CASES, scale=SCALE.tolist(), n_obs=N_OBS,
                git=be.git_info(be.repo_root))
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=2, default=str)

    W = np.zeros((8, N_OBS))
    base_ret, base_m = runner.evaluate(W, TRAIN_CASES)
    best = (base_ret, W.copy())
    rows = [dict(iter=0, ret=base_ret, n_ok=sum(not m["fell"] for m in base_m), wall_s=0.0, sigma_r=0.0, best_dir=base_ret)]
    print(f"iter 0 (baseline): return {base_ret:.3f}, survived {rows[0]['n_ok']}/{len(TRAIN_CASES)}", flush=True)
    for it in range(1, args.iters + 1):
        tic = time.perf_counter()
        deltas = rng.normal(size=(args.dirs, 8, N_OBS))
        r_plus = np.array([runner.evaluate(W + args.nu * d, TRAIN_CASES)[0] for d in deltas])
        r_minus = np.array([runner.evaluate(W - args.nu * d, TRAIN_CASES)[0] for d in deltas])
        top = np.argsort(-np.maximum(r_plus, r_minus))[: args.top]
        sigma = float(np.std(np.concatenate([r_plus[top], r_minus[top]]))) or 1.0
        W = W + args.alpha / (args.top * sigma) * np.einsum("k,kij->ij", r_plus[top] - r_minus[top], deltas[top])
        ret, ms = runner.evaluate(W, TRAIN_CASES)
        n_ok = sum(not m["fell"] for m in ms)
        if ret > best[0]:
            best = (ret, W.copy())
        np.save(os.path.join(out, f"W_iter{it:02d}.npy"), W)
        np.save(os.path.join(out, "W_best.npy"), best[1])
        rows.append(dict(iter=it, ret=ret, n_ok=n_ok, wall_s=time.perf_counter() - tic, sigma_r=sigma,
                         best_dir=float(np.max(np.maximum(r_plus, r_minus)))))
        be.write_csv(os.path.join(out, "log.csv"), rows)
        falls = ", ".join(f"{c[1]}@{m['fall_time_s']:.1f}s" for c, m in zip(TRAIN_CASES, ms) if m["fell"])
        print(f"iter {it}: return {ret:.3f} (best direction {rows[-1]['best_dir']:.3f}), survived "
              f"{n_ok}/{len(TRAIN_CASES)}{' [fell: ' + falls + ']' if falls else ''}, |W| {np.linalg.norm(W):.2f}, "
              f"{rows[-1]['wall_s']:.0f}s", flush=True)
    print(f"Best return {best[0]:.3f} (baseline {base_ret:.3f}). Results in {out}")


def evaluate(args):
    """The trained W vs the baseline on flat (all 7 scenarios, no-degradation check) and the curb scene."""
    runner = Runner(args.state)
    W = np.load(args.weights)
    cases = [("flat", s) for s in be.SCENARIOS] + [("steps", s) for s in be.SCENARIOS if s != "vx_fast"]
    for scene, scen in cases:
        r0, m0 = runner.rollout(np.zeros_like(W), scene, scen)
        r1, m1 = runner.rollout(W, scene, scen)
        f = lambda m: f"FELL@{m['fall_time_s']:.1f}s" if m["fell"] else "ok"
        print(f"{scene:6s} {scen:9s} baseline {f(m0):10s} rmse_v {m0['rmse_v_steady']:.3f} ret {r0:.3f} | "
              f"residual {f(m1):10s} rmse_v {m1['rmse_v_steady']:.3f} ret {r1:.3f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("train")
    t.add_argument("--iters", type=int, default=15)
    t.add_argument("--dirs", type=int, default=6)
    t.add_argument("--top", type=int, default=3)
    t.add_argument("--alpha", type=float, default=0.02)
    t.add_argument("--nu", type=float, default=0.05)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--state", default="nominal", choices=be.STATE_SOURCES)
    t.add_argument("--tag", default="curbs")
    e = sub.add_parser("eval")
    e.add_argument("weights")
    e.add_argument("--state", default="nominal", choices=be.STATE_SOURCES)
    args = p.parse_args()
    train(args) if args.mode == "train" else evaluate(args)


if __name__ == "__main__":
    main()
