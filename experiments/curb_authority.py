"""Can any task-space residual channel change the curb failure at all? (Phase 3, step 3 follow-up)

The ARS run on the curb cases left the fall times of vx_step (5.4-5.5 s) and backward (4.2 s)
unchanged in every evaluation. This probe drives each residual channel at its full scale
(residual_ars.SCALE), both signs, for the whole episode, on the unperturbed curb scene, and
prints the fall time, so we see whether the action space has any authority over the failure.

Usage (from the mpx_tita root):  pixi run python experiments/curb_authority.py   (~10 min)
"""
import os

import numpy as np

import baseline_eval as be
import robustness_eval as re_
from mpx.utils.mpc_wrapper_dfcip import BatchedMPCControllerWrapper
from residual_ars import SCALE

SCENARIOS = ["vx_step", "backward"]


def main():
    os.chdir(be.repo_root)
    mpc = BatchedMPCControllerWrapper(be.config, n_env=1)
    model = re_.load_model("steps")
    layout = BatchedMPCControllerWrapper.RESIDUAL_LAYOUT
    for sc in SCENARIOS:
        m, _ = be.run_episode(mpc, model, be.SCENARIOS[sc], state="nominal")
        print(f"{sc:9s} baseline              {'FELL@%.2fs' % m['fall_time_s'] if m['fell'] else 'ok'}", flush=True)
        for ch, name in enumerate(layout):
            for sign in (+1, -1):
                r = np.zeros(8); r[ch] = sign * SCALE[ch]
                m, _ = be.run_episode(mpc, model, be.SCENARIOS[sc], residual_fn=lambda *a, r=r: r,
                                      state="nominal")
                res = f"FELL@{m['fall_time_s']:.2f}s" if m["fell"] else "ok"
                print(f"{sc:9s} {name:14s} {sign * SCALE[ch]:+5.1f}  {res:10s} max tilt {m['max_tilt_deg']:5.1f}deg",
                      flush=True)


if __name__ == "__main__":
    main()
