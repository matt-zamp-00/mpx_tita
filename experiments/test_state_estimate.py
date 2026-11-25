"""Check of the nominal-model state estimate (Phase 3, step 2), without running the controller.

  1. unperturbed plant: the estimate equals the simulator's truth exactly;
  2. perturbations that do not change the mass (strength, friction, damping): also exact;
  3. payload: only the CoM and its velocity differ; prints by how much.

Usage (from the mpx_tita root):  pixi run python experiments/test_state_estimate.py
"""
import os

import mujoco
import numpy as np

import baseline_eval as be
import perturbations as P
from mpx.utils import sim as sim_utils

NAMES = ("pcom", "vcom", "centers", "Rs", "radii", "feet_vel")


def compare(nominal, pert, rng):
    model = pert.apply_model(nominal)
    data, ndata = mujoco.MjData(model), mujoco.MjData(nominal)
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, be.config.base_body_name)
    contacts = sim_utils.geom_ids(model, be.config.contact_frame)
    be._reset_to_initial_state(model, data)
    q0 = data.qpos.copy()
    worst = {k: 0.0 for k in NAMES}
    for i in range(21):   # the standing pose at rest, then random poses / velocities around it
        data.qpos[:] = q0 + (i > 0) * 0.05 * rng.normal(size=model.nq) * (np.arange(model.nq) >= 7)
        data.qvel[:] = (i > 0) * 0.3 * rng.normal(size=model.nv)
        mujoco.mj_forward(model, data)
        true = be._gather_raw_state(model, data, base, contacts)
        est = be._gather_raw_state(nominal, be.nominal_estimate(nominal, ndata, data), base, contacts)
        for k, a, b in zip(NAMES, true, est):
            worst[k] = max(worst[k], float(np.max(np.abs(np.asarray(a) - np.asarray(b)))))
        if i == 0:
            standing = np.asarray(true[0]) - np.asarray(est[0])
    return worst, standing


def main():
    os.chdir(be.repo_root)
    nominal = mujoco.MjModel.from_xml_path(os.path.join(be.repo_root, "mpx/data/tita/scene_flat.xml"))
    rng = np.random.default_rng(0)

    w, _ = compare(nominal, P.Perturbation(), rng)
    assert all(v == 0.0 for v in w.values()), w
    print("1. unperturbed: estimate == truth (exact)")

    w, _ = compare(nominal, P.Perturbation(strength_scale=0.8, friction_scale=0.5, joint_damping=1.0), rng)
    assert all(v == 0.0 for v in w.values()), w
    print("2. strength/friction/damping: estimate == truth (exact)")

    for kg in (2.0, 5.0, 7.5):
        w, st = compare(nominal, P.Perturbation(payload_kg=kg), rng)
        assert max(w[k] for k in NAMES[2:]) < 1e-12, w   # wheel poses are kinematic (rounding only)
        print(f"3. payload {kg} kg: CoM error standing (true - estimate) x/y/z "
              f"{1e3 * st[0]:+.1f}/{1e3 * st[1]:+.1f}/{1e3 * st[2]:+.1f} mm; over random poses max "
              f"{1e3 * w['pcom']:.1f} mm, CoM velocity {w['vcom']:.3f} m/s; wheel poses exact")


if __name__ == "__main__":
    main()
