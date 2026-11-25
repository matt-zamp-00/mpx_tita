"""Authority of each task-space residual channel (Phase 3): a residual of 2 on one channel for 0.5 s
during stand on flat ground, compared with the baseline over [3, 4] s.

Usage (from the mpx_tita root):  pixi run python experiments/residual_authority.py     (~1 min)
"""
import os

import numpy as np

import baseline_eval as be

W = be.mpc_wrapper_dfcip
os.chdir(be.repo_root)
model = be.mujoco.MjModel.from_xml_path("mpx/data/tita/scene_flat.xml"); model.opt.timestep = be.config.dt_sim
mpc = be.mpc_wrapper_dfcip.BatchedMPCControllerWrapper(be.config, n_env=1)
sc = dict(be.SCENARIOS["stand"]); sc["duration"] = 4.5
def probe(ch, amp):
    def f(t, x0, data, s):
        r = np.zeros(8)
        if 3.0 <= t < 3.5: r[ch] = amp
        return r
    return f
m0, t0 = be.run_episode(mpc, model, sc)
w = (t0["t"] >= 3.0) & (t0["t"] < 4.0)
for ch, name in enumerate(W.BatchedMPCControllerWrapper.RESIDUAL_LAYOUT):
    amp = 2.0
    m, t = be.run_episode(mpc, model, sc, residual_fn=probe(ch, amp))
    n = min(len(t0["t"]), len(t["t"])); ww = w[:n]
    dtau = np.abs(t["tau"][:n] - t0["tau"][:n])[ww]
    print(f"{name:15s} amp {amp}: mean|dtau| {dtau.mean():.3f} max {dtau.max():.2f} N*m | max dv {np.abs(t['v'][:n]-t0['v'][:n])[ww].max():.3f} m/s "
          f"dw {np.abs(t['w'][:n]-t0['w'][:n])[ww].max():.3f} rad/s dh {np.abs(t['com_h'][:n]-t0['com_h'][:n])[ww].max()*100:.2f} cm "
          f"dtilt {np.degrees(np.abs(t['tilt'][:n]-t0['tilt'][:n])[ww].max()):.2f} deg {'FELL' if m['fell'] else ''}", flush=True)
