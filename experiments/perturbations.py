"""Perturbations of the simulated robot for the robustness study (Phase 2).

Only the *simulated plant* is perturbed: the MPC + WBC keep their nominal model (config_tita /
tita.xml). This creates the model mismatch the residual policy will have to compensate.

A Perturbation is applied in three places of an episode (baseline_eval.run_episode):
  1. apply_model(model)       -> perturbed copy of the MjModel (payload, motors, friction, damping)
  2. apply_initial_state(...) -> shifted / rotated start pose, resting on the terrain
  3. every physics step       -> actuation latency (delayed torques) and an external push

Perturbation() with the defaults is the nominal case and leaves the episode unchanged.
"""
import copy
import dataclasses

import mujoco
import numpy as np

LEG_JOINTS = [f"joint_{side}_leg_{i}" for side in ("left", "right") for i in (1, 2, 3)]


@dataclasses.dataclass
class Perturbation:
    payload_kg: float = 0.0        # extra mass on base_link at its CoM (inertia scaled with the mass)
    strength_scale: float = 1.0    # actuator gear scale (all motors): torque delivered = scale * command
    friction_scale: float = 1.0    # sliding friction of every geom (contact uses the max of the pair)
    joint_damping: float = 0.0     # N*m*s/rad added to the leg joints (not the wheels)
    latency_ms: float = 0.0        # delay between the WBC torque and its application
    push_force_n: float = 0.0      # magnitude of a horizontal force on base_link
    push_dir_rad: float = 0.0      # direction in the base frame at push time (0 = forward, pi/2 = left)
    push_time_s: float = 3.0
    push_duration_s: float = 0.1
    init_dx: float = 0.0           # start pose offset in the world frame (m)
    init_dy: float = 0.0
    init_yaw: float = 0.0          # rad, about the base

    def as_dict(self):
        return dataclasses.asdict(self)

    # -- 1. model ----------------------------------------------------------------------------
    def apply_model(self, model):
        m = copy.deepcopy(model)
        base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if self.payload_kg:
            scale = (m.body_mass[base] + self.payload_kg) / m.body_mass[base]
            m.body_mass[base] *= scale
            m.body_inertia[base] *= scale
            # body_subtreemass is precomputed at compile time and used by mj_comPos/mj_subtreeVel
            # (subtree_com = sum(m_i x_i) / subtreemass): recompute it, else the simulator's CoM
            # is scaled by (M + payload) / M in world coordinates. Bug until 2026-11-25: Phase 2
            # payload results fed the controller this wrong CoM (see thesis.md).
            m.body_subtreemass[:] = m.body_mass
            for i in range(m.nbody - 1, 0, -1):
                m.body_subtreemass[m.body_parentid[i]] += m.body_subtreemass[i]
        if self.strength_scale != 1.0:
            m.actuator_gear[:, 0] *= self.strength_scale
        if self.friction_scale != 1.0:
            m.geom_friction[:, 0] *= self.friction_scale
        if self.joint_damping:
            for name in LEG_JOINTS:
                j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
                m.dof_damping[m.jnt_dofadr[j]] += self.joint_damping
        return m

    # -- 2. initial state --------------------------------------------------------------------
    def apply_initial_state(self, model, data):
        """Shift by (init_dx, init_dy) and rotate by init_yaw about the base. The robot is lifted or
        lowered by the terrain height difference between the old and new base xy (raycast with the
        robot moved out of the way), so it starts resting on uneven terrain as on the flat start."""
        if not (self.init_dx or self.init_dy or self.init_yaw):
            return
        x0, y0, z0 = data.qpos[0:3].copy()
        x1, y1 = x0 + self.init_dx, y0 + self.init_dy
        data.qpos[2] = 100.0
        mujoco.mj_forward(model, data)
        dz = ground_height(model, data, x1, y1) - ground_height(model, data, x0, y0)
        data.qpos[0:3] = [x1, y1, z0 + dz]
        q_yaw = np.array([np.cos(self.init_yaw / 2), 0.0, 0.0, np.sin(self.init_yaw / 2)])
        q = np.empty(4)
        mujoco.mju_mulQuat(q, q_yaw, data.qpos[3:7].copy())
        data.qpos[3:7] = q
        mujoco.mj_forward(model, data)

    # -- 3. per step ---------------------------------------------------------------------------
    def latency_steps(self, dt):
        return int(round(self.latency_ms * 1e-3 / dt))

    def push_wrench(self, t, base_xmat):
        """xfrc_applied row for base_link at time t (zeros outside the push window)."""
        f = np.zeros(6)
        if self.push_force_n and self.push_time_s <= t < self.push_time_s + self.push_duration_s:
            d_base = np.array([np.cos(self.push_dir_rad), np.sin(self.push_dir_rad), 0.0])
            d_world = base_xmat.reshape(3, 3) @ d_base
            d_world[2] = 0.0
            f[:3] = self.push_force_n * d_world / max(np.linalg.norm(d_world), 1e-9)
        return f


def ground_height(model, data, x, y):
    """z of the first geom hit by a ray cast down from z = 5 m at (x, y)."""
    geomid = np.array([-1], dtype=np.int32)
    dist = mujoco.mj_ray(model, data, np.array([x, y, 5.0]), np.array([0.0, 0.0, -1.0]),
                         None, 1, -1, geomid)
    return 5.0 - dist if dist >= 0 else 0.0


# ------------------------------------------------------------------------------------------------
# Study design
# ------------------------------------------------------------------------------------------------

# One-factor-at-a-time sweep: the nominal value first.
SWEEP = {
    "payload_kg":     [0.0, 2.5, 5.0, 7.5],
    "strength_scale": [1.0, 0.9, 0.8, 0.7],
    "friction_scale": [1.0, 0.7, 0.5, 0.3],
    "joint_damping":  [0.0, 0.5, 1.0, 2.0],
    "latency_ms":     [0.0, 4.0, 10.0, 20.0],
    "push_force_n":   [0.0, 50.0, 100.0, 150.0],
}

# Domain randomization ranges (uniform), also the training ranges for the residual (Phase 5).
# "main": the realistic benchmark (payload 0-2 kg, decided 2026-11-25). Payload and pushes beyond it
#         are evaluated separately as hard axes (HARD_AXES, one factor at a time).
# "wide": the first Phase 2 study (payload 0-5 kg), dominated by payload; kept for reference.
_COMMON_RANGES = {
    "strength_scale": (0.75, 1.0),
    "friction_scale": (0.4, 1.0),
    "joint_damping":  (0.0, 1.0),
    "latency_ms":     (0.0, 10.0),
    "push_force_n":   (0.0, 100.0),
    "init_dx":        (-0.3, 0.3),
    "init_dy":        (-0.3, 0.3),
    "init_yaw":       (-0.3, 0.3),
}
RANGES = {
    "main": {"payload_kg": (0.0, 2.0), **_COMMON_RANGES},
    "wide": {"payload_kg": (0.0, 5.0), **_COMMON_RANGES},
}
RANDOM_RANGES = RANGES["main"]

# Hard axes: beyond the main benchmark, one factor at a time (robustness_eval.py sweep --factors).
HARD_AXES = ["payload_kg", "push_force_n"]


def sample(rng, duration_s, ranges=RANDOM_RANGES):
    """One random Perturbation. The push comes at a random time in [2 s, duration - 2 s] and a
    random direction."""
    kw = {k: float(rng.uniform(*v)) for k, v in ranges.items()}
    kw["push_dir_rad"] = float(rng.uniform(-np.pi, np.pi))
    kw["push_time_s"] = float(rng.uniform(2.0, max(2.0, duration_s - 2.0)))
    return Perturbation(**kw)
