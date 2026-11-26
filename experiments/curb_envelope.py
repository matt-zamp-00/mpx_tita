"""Envelope of curb heights and speeds the controller can drive over (benchmark v2 design, 2026-11-26).

One curb up at x = 1.2 m (platform step_1 of scene_steps.xml, its height set here), driven straight at a
constant commanded speed; no other perturbation. Success = no fall within the episode.

Result 2026-11-26 (original and tuned controller alike, nominal state): only 1 cm at 1 m/s passes; every curb
fails at 0.3 and 0.5 m/s (the wheel stalls at the edge, then the legs fold and the speed runs away).

Usage (from the mpx_tita root):  pixi run python experiments/curb_envelope.py [--set ...] [--state nominal]
"""
import argparse
import os

import mujoco
import numpy as np

import baseline_eval as be

HEIGHTS = [0.01, 0.02, 0.03, 0.05]
SPEEDS = [0.3, 0.5, 1.0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", nargs="+", default=[])
    parser.add_argument("--state", default="nominal", choices=be.STATE_SOURCES)
    args = parser.parse_args()
    be.apply_overrides(be.config, args.set)
    os.chdir(be.repo_root)
    mpc = be.mpc_wrapper_dfcip.BatchedMPCControllerWrapper(be.config, n_env=1)
    base = mujoco.MjModel.from_xml_path("mpx/data/tita/scene_steps.xml")
    base.opt.timestep = be.config.dt_sim
    g = mujoco.mj_name2id(base, mujoco.mjtObj.mjOBJ_GEOM, "step_1")
    print(f"{'height':>7s} " + " ".join(f"{v:>6.1f}m/s" for v in SPEEDS))
    for h in HEIGHTS:
        model = mujoco.MjModel.from_xml_path("mpx/data/tita/scene_steps.xml")
        model.opt.timestep = be.config.dt_sim
        model.geom_size[g, 2] = h / 2
        model.geom_pos[g, 2] = h / 2
        cells = []
        for v in SPEEDS:
            dur = 1.0 + 2.4 / v + 1.5          # reach the top of the platform, then settle
            sc = dict(duration=dur, segments=[(0.0, 0.0, 0.0, be.H0), (1.0, v, 0.0, be.H0)])
            m, _ = be.run_episode(mpc, model, sc, state=args.state)
            cells.append(f"FELL {m['fall_time_s']:.1f}" if m["fell"] else "    ok   ")
        print(f"{100 * h:>5.0f}cm " + " ".join(f"{c:>9s}" for c in cells), flush=True)


if __name__ == "__main__":
    main()
