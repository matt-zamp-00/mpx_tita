"""Robustness of the TITA MPC + WBC baseline under perturbations (Phase 2 of the residual-RL workflow).

Two modes, both with the scenarios and metrics of baseline_eval.py:

  sweep   one factor at a time (perturbations.SWEEP), the others nominal. Shows which mismatch
          hurts the baseline and from which magnitude. Default: flat scene, so terrain does not
          mix in.
  random  every factor sampled at once (perturbations.RANDOM_RANGES) for N seeds. The
          perturbation depends only on (seed, scenario), so every scene, and later the
          MPC + residual controller, is evaluated on exactly the same perturbed cases (paired
          comparison).

Usage (from the mpx_tita root):
    pixi run python experiments/robustness_eval.py sweep                                # ~20 min
    pixi run python experiments/robustness_eval.py sweep --factors latency_ms push_force_n
    pixi run python experiments/robustness_eval.py random --seeds 10        # main benchmark, ~1 h
    pixi run python experiments/robustness_eval.py sweep --hard             # payload/push axes, ~10 min
    pixi run python experiments/robustness_eval.py random --ranges wide     # first study (payload 0-5 kg)

Output: experiments/results/<timestamp>_<mode>_<tag>/{meta.json, metrics.csv, summary.csv}
"""
import argparse
import datetime
import json
import os
import time

import numpy as np

import baseline_eval as be
import perturbations as P

SWEEP_SCENARIOS = ["stand", "vx_step", "turn", "arc"]


STATE = "true"   # set from --state in main()


def run_and_print(mpc, model, scene, scenario, pert, label):
    tic = time.perf_counter()
    metrics, _ = be.run_episode(mpc, model, be.SCENARIOS[scenario], pert, state=STATE)
    status = f"FELL at {metrics['fall_time_s']:.2f}s" if metrics["fell"] else "ok"
    print(f"[{scene:6s} {scenario:9s} {label:26s}] {status:14s} rmse_v {metrics['rmse_v_steady']:.3f} "
          f"rmse_w {metrics['rmse_w_steady']:.3f} tilt {metrics['max_tilt_deg']:5.1f}deg "
          f"({time.perf_counter() - tic:.0f}s wall)", flush=True)
    return metrics


def load_model(scene):
    model = be.mujoco.MjModel.from_xml_path(
        os.path.join(be.repo_root, f"mpx/data/tita/scene_{scene}.xml"))
    model.opt.timestep = be.config.dt_sim
    return model


def mode_sweep(args, mpc):
    rows = []
    for scene in args.scenes:
        model = load_model(scene)
        cases = [("nominal", None)] + [(f, v) for f in args.factors for v in P.SWEEP[f][1:]]
        for factor, value in cases:
            pert = P.Perturbation() if factor == "nominal" else P.Perturbation(**{factor: value})
            for scenario in args.scenarios:
                label = "nominal" if factor == "nominal" else f"{factor}={value:g}"
                m = run_and_print(mpc, model, scene, scenario, pert, label)
                rows.append(dict(scene=scene, scenario=scenario, factor=factor,
                                 value=value if value is not None else float("nan"), **m))

    # Summary: per (scene, factor, value), success over the scenarios.
    summary = []
    for scene in args.scenes:
        for factor in ["nominal"] + args.factors:
            values = [None] if factor == "nominal" else P.SWEEP[factor][1:]
            for v in values:
                rs = [r for r in rows if r["scene"] == scene and r["factor"] == factor
                      and (v is None or r["value"] == v)]
                ok = [r for r in rs if not r["fell"]]
                summary.append(dict(
                    scene=scene, factor=factor, value=v if v is not None else float("nan"),
                    n=len(rs), n_ok=len(ok),
                    failed=" ".join(r["scenario"] for r in rs if r["fell"]),
                    rmse_v_steady_mean=float(np.nanmean([r["rmse_v_steady"] for r in ok])) if ok else float("nan"),
                    rmse_w_steady_mean=float(np.nanmean([r["rmse_w_steady"] for r in ok])) if ok else float("nan"),
                    max_tilt_deg_max=float(max(r["max_tilt_deg"] for r in rs)),
                ))
    print("\nSweep summary (success over the scenarios, tracking over the runs that did not fall):")
    print(f"{'scene':6s} {'factor':15s} {'value':>6s} {'ok':>5s}  {'rmse_v':>7s} {'rmse_w':>7s} "
          f"{'tilt':>6s}  failed")
    for s in summary:
        print(f"{s['scene']:6s} {s['factor']:15s} {s['value']:>6g} {s['n_ok']:>2d}/{s['n']:<2d}  "
              f"{s['rmse_v_steady_mean']:7.3f} {s['rmse_w_steady_mean']:7.3f} "
              f"{s['max_tilt_deg_max']:6.1f}  {s['failed']}")
    return rows, summary


def mode_random(args, mpc):
    rows = []
    for scene in args.scenes:
        model = load_model(scene)
        for i, scenario in enumerate(args.scenarios):
            if not be.in_benchmark(args.bench, scene, scenario):
                continue
            for seed in range(args.seeds):
                rng = np.random.default_rng([args.base_seed, seed, list(be.SCENARIOS).index(scenario)])
                pert = P.sample(rng, be.SCENARIOS[scenario]["duration"], P.RANGES[args.ranges])
                m = run_and_print(mpc, model, scene, scenario, pert, f"seed {seed}")
                rows.append(dict(scene=scene, scenario=scenario, seed=seed, **m,
                                 **{f"p_{k}": v for k, v in pert.as_dict().items()}))
    summary = be.summarize([dict(r, repeat=r["seed"]) for r in rows])
    be.print_summary(summary)
    print("\nSuccess rate per scene:")
    for scene in args.scenes:
        rs = [r for r in rows if r["scene"] == scene]
        print(f"  {scene:6s} {sum(not r['fell'] for r in rs):>3d}/{len(rs)}")
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["sweep", "random"])
    parser.add_argument("--scenes", nargs="+", choices=be.SCENES)
    parser.add_argument("--scenarios", nargs="+", choices=list(be.SCENARIOS))
    parser.add_argument("--factors", nargs="+", default=list(P.SWEEP), choices=list(P.SWEEP),
                        help="sweep mode: factors to vary")
    parser.add_argument("--seeds", type=int, default=5, help="random mode: perturbations per scenario")
    parser.add_argument("--base-seed", type=int, default=0, help="random mode: changes the whole set")
    parser.add_argument("--ranges", default="main", choices=list(P.RANGES),
                        help="random mode: perturbations.RANGES preset (main = payload 0-2 kg)")
    parser.add_argument("--bench", default="v2", choices=list(be.BENCHMARKS),
                        help="random mode: scenes and scene/scenario cases (baseline_eval.BENCHMARKS); "
                             "v1 = Phase 2 benchmark, v2 = realistic envelope (default since 2026-11-26)")
    parser.add_argument("--hard", action="store_true",
                        help="sweep mode: only the hard axes (perturbations.HARD_AXES), all scenarios")
    parser.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE",
                        help="config overrides, e.g. N=24 terminal_stability=dcm W_mpc.w_stab=1e6")
    parser.add_argument("--state", default="true", choices=be.STATE_SOURCES,
                        help="controller state: simulator truth (Phases 1-2) or nominal-model estimate (real robot)")
    parser.add_argument("--tag", default=None, help="default: baseline_<ranges> / baseline_hard")
    parser.add_argument("--out", default=os.path.join(be.dir_path, "results"))
    args = parser.parse_args()
    if args.mode == "sweep" and args.hard:
        args.factors = list(P.HARD_AXES)
        if args.scenarios is None:
            args.scenarios = list(be.SCENARIOS)
    if args.tag is None:
        args.tag = ("baseline_hard" if args.hard else "baseline") if args.mode == "sweep" else f"baseline_{args.ranges}_{args.bench}"
    if args.scenes is None:
        args.scenes = ["flat"] if args.mode == "sweep" else be.BENCHMARKS[args.bench]["scenes"]
    if args.scenarios is None:
        args.scenarios = SWEEP_SCENARIOS if args.mode == "sweep" else list(be.SCENARIOS)

    global STATE
    STATE = args.state
    overrides = be.apply_overrides(be.config, args.set)
    os.chdir(be.repo_root)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out, f"{stamp}_{args.mode}_{args.tag}")
    os.makedirs(out_dir, exist_ok=True)
    meta = dict(date=stamp, mode=args.mode, tag=args.tag, git=be.git_info(be.repo_root),
                scenes=args.scenes, scenarios=args.scenarios, factors=args.factors,
                seeds=args.seeds, base_seed=args.base_seed, overrides=overrides, state=args.state, bench=args.bench,
                sweep=P.SWEEP, ranges=args.ranges, random_ranges=P.RANGES[args.ranges],
                nominal_perturbation=P.Perturbation().as_dict(),
                jax_devices=[str(d) for d in be.jax.devices()])
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=float)
    if meta["git"]["dirty"]:
        print("WARNING: mpx_tita has uncommitted changes; these results are not reproducible "
              "from the recorded commit alone.")

    print("Building and compiling the MPC + WBC...")
    mpc = be.mpc_wrapper_dfcip.BatchedMPCControllerWrapper(be.config, n_env=1)
    rows, summary = (mode_sweep if args.mode == "sweep" else mode_random)(args, mpc)

    be.write_csv(os.path.join(out_dir, "metrics.csv"), rows)
    be.write_csv(os.path.join(out_dir, "summary.csv"), summary)
    print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main()
