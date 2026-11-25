"""Compare baseline-controller variants on a small tuning set (baseline improvement step).

The tuning set is deliberately separate from the benchmark (robustness_eval.py random --seeds 10
and sweep --hard), so the improved baseline is not tuned on the cases it is evaluated on.
It mixes nominal cases (tracking must not degrade) with the failure modes found in Phase 2
(pushes, payload, terrain).

Usage (from the mpx_tita root):
    pixi run python experiments/tune_baseline.py                       # all VARIANTS below
    pixi run python experiments/tune_baseline.py --variants base dcm_1e5

Each variant is a set of config overrides (same syntax as --set in the eval scripts).
Output: experiments/results/<timestamp>_tune/{meta.json, metrics.csv, summary.csv}
"""
import argparse
import datetime
import importlib
import json
import os
import time

import numpy as np

import baseline_eval as be
import perturbations as P

TUNING_SET = [
    # (scene, scenario, perturbation, label)
    ("flat",   "vx_step",  P.Perturbation(),                                  "nominal"),
    ("flat",   "vx_fast",  P.Perturbation(),                                  "nominal"),
    ("flat",   "turn",     P.Perturbation(),                                  "nominal"),
    ("flat",   "arc",      P.Perturbation(),                                  "nominal"),
    ("rough",  "vx_step",  P.Perturbation(),                                  "terrain"),
    ("perlin", "vx_step",  P.Perturbation(),                                  "terrain"),
    ("stairs", "vx_step",  P.Perturbation(),                                  "terrain"),
    ("flat",   "stand",    P.Perturbation(push_force_n=150.0),                "push fwd 150"),
    ("flat",   "stand",    P.Perturbation(push_force_n=150.0, push_dir_rad=np.pi), "push back 150"),
    ("flat",   "vx_step",  P.Perturbation(push_force_n=100.0),                "push fwd 100"),
    ("flat",   "stand",    P.Perturbation(payload_kg=7.5),                    "payload 7.5"),
    ("flat",   "arc",      P.Perturbation(payload_kg=5.0),                    "payload 5"),
]

VARIANTS = {
    "base":         [],
    "com_1e6":      ["W_mpc.w_stab=1e6"],
    "dcm_1e4":      ["terminal_stability=dcm"],
    "dcm_1e5":      ["terminal_stability=dcm", "W_mpc.w_stab=1e5"],
    "dcm_1e6":      ["terminal_stability=dcm", "W_mpc.w_stab=1e6"],
    "dcm_1e5_N24":  ["terminal_stability=dcm", "W_mpc.w_stab=1e5", "N=24"],
    "dcm_1e5_dt50": ["terminal_stability=dcm", "W_mpc.w_stab=1e5", "dt_mpc=0.05"],
    # round 2 (2026-11-25): attribution and neighbours of dcm_1e5_dt50
    "com_dt50":         ["dt_mpc=0.05"],
    "dcm_1e4_dt50":     ["terminal_stability=dcm", "dt_mpc=0.05"],
    "dcm_1e6_dt50":     ["terminal_stability=dcm", "W_mpc.w_stab=1e6", "dt_mpc=0.05"],
    "dcm_1e5_dt50_it2": ["terminal_stability=dcm", "W_mpc.w_stab=1e5", "dt_mpc=0.05", "mpc_iterations=2"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--out", default=os.path.join(be.dir_path, "results"))
    args = parser.parse_args()

    os.chdir(be.repo_root)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out, f"{stamp}_tune")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(dict(date=stamp, git=be.git_info(be.repo_root),
                       variants={k: VARIANTS[k] for k in args.variants},
                       tuning_set=[(s, sc, p.as_dict(), l) for s, sc, p, l in TUNING_SET]),
                  f, indent=2, default=float)

    models = {}
    rows, summary = [], []
    for name in args.variants:
        importlib.reload(be.config)              # fresh nominal config for every variant
        be.apply_overrides(be.config, VARIANTS[name])
        mpc = be.mpc_wrapper_dfcip.BatchedMPCControllerWrapper(be.config, n_env=1)
        cells, oks, mpc_ms = [], [], []
        for scene, scenario, pert, label in TUNING_SET:
            if scene not in models:
                models[scene] = be.mujoco.MjModel.from_xml_path(
                    os.path.join(be.repo_root, f"mpx/data/tita/scene_{scene}.xml"))
            model = models[scene]
            model.opt.timestep = be.config.dt_sim
            m, _ = be.run_episode(mpc, model, be.SCENARIOS[scenario], pert)
            rows.append(dict(variant=name, scene=scene, scenario=scenario, case=label, **m))
            oks.append(not m["fell"])
            mpc_ms.append(m["mpc_ms_mean"])
            cells.append(f"{scene[:2]}/{scenario}/{label}: " +
                         (f"FELL {m['fall_time_s']:.1f}" if m["fell"] else f"{m['rmse_v_steady']:.3f}"))
        nominal = [r for r in rows if r["variant"] == name and r["case"] == "nominal" and not r["fell"]]
        s = dict(variant=name, overrides=" ".join(VARIANTS[name]), ok=int(sum(oks)), n=len(oks),
                 nominal_ok=sum(1 for r in rows if r["variant"] == name and r["case"] == "nominal" and not r["fell"]),
                 nominal_rmse_v=float(np.mean([r["rmse_v_steady"] for r in nominal])) if nominal else float("nan"),
                 nominal_rmse_w=float(np.mean([r["rmse_w_steady"] for r in nominal])) if nominal else float("nan"),
                 mpc_ms=float(np.nanmean(mpc_ms)))
        summary.append(s)
        print(f"\n== {name} ({s['overrides'] or 'original'}): {s['ok']}/{s['n']} ok, nominal {s['nominal_ok']}/4 "
              f"rmse_v {s['nominal_rmse_v']:.3f} rmse_w {s['nominal_rmse_w']:.3f}, mpc {s['mpc_ms']:.1f} ms",
              flush=True)
        for c in cells:
            print("   " + c, flush=True)

    be.write_csv(os.path.join(out_dir, "metrics.csv"), rows)
    be.write_csv(os.path.join(out_dir, "summary.csv"), summary)
    print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main()
