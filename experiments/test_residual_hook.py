"""Regression test of the residual hook (Phase 3, step 1).

Checks:
  1. layout: a residual changes only its slots of the WBC `desired` vector, with the right frame;
  2. zero residual == no residual: `desired` is bit-identical, and flat episodes give the same
     metrics and torques within the run-to-run noise of JAX on the GPU (~0.006 N*m between two
     plain baseline runs, measured 2026-11-25; the episodes are not bitwise reproducible);
  3. the baseline is unchanged: those episodes match the Phase 1 reference (results/20261123_baseline_ref_x5);
  4. the hook is live: a constant com_acc_z residual changes the torques well beyond the noise.
     (The steady-state tracking error barely moves: the WBC PD + integral terms reject a constant
     feedforward offset. A useful residual has to act on the transients.)

Usage (from the mpx_tita root):  pixi run python experiments/test_residual_hook.py     (~2 min)
"""
import os

import jax.numpy as jnp
import numpy as np

import baseline_eval as be

TAU_NOISE = 0.05   # N*m, tolerance on per-step torques (run-to-run noise is ~0.006)

# Phase 1 reference (5 repeats, identical across repeats): flat rmse_v_steady
REFERENCE = {"vx_step": 0.0149, "arc": 0.0146}


def check_layout(mpc):
    idx = mpc._index_variables_qp
    desired = jnp.zeros((1, mpc._desired_size))
    theta = np.pi / 3
    x0 = jnp.zeros(13).at[10].set(theta)
    r = jnp.arange(1.0, 9.0)  # 1..8
    d = np.asarray(mpc._apply_residual_jit(desired, x0, r))[0]
    h = np.array([np.cos(theta), np.sin(theta), 0.0])
    expected = np.zeros_like(d)
    expected[idx['com_acc']:idx['com_acc'] + 3] = [1, 2, 3]
    expected[idx['base_alpha']:idx['base_alpha'] + 3] = [4, 5, 6]
    expected[idx['lwheel_acc']:idx['lwheel_acc'] + 3] = 7 * h
    expected[idx['rwheel_acc']:idx['rwheel_acc'] + 3] = 8 * h
    assert np.allclose(d, expected), "residual written to the wrong slots"
    print("1. layout: ok")
    rnd = jnp.asarray(np.random.default_rng(0).normal(size=(1, mpc._desired_size)))
    assert np.array_equal(np.asarray(mpc._apply_residual_jit(rnd, x0, jnp.zeros(8))), np.asarray(rnd))
    print("2. zero residual leaves `desired` bit-identical")


def main():
    os.chdir(be.repo_root)
    model = be.mujoco.MjModel.from_xml_path(os.path.join(be.repo_root, "mpx/data/tita/scene_flat.xml"))
    model.opt.timestep = be.config.dt_sim
    mpc = be.mpc_wrapper_dfcip.BatchedMPCControllerWrapper(be.config, n_env=1)

    check_layout(mpc)

    zero = lambda t, x0, data, s: np.zeros(8)
    for name, ref in REFERENCE.items():
        sc = be.SCENARIOS[name]
        m0, t0 = be.run_episode(mpc, model, sc)
        m1, t1 = be.run_episode(mpc, model, sc, residual_fn=zero)
        assert not m0["fell"] and not m1["fell"], f"{name}: baseline fell"
        assert t0["tau"].shape == t1["tau"].shape, f"{name}: different episode lengths"
        dtau = float(np.max(np.abs(t0["tau"] - t1["tau"])))
        assert dtau < TAU_NOISE, f"{name}: zero residual changed the torques by {dtau:.3g} N*m"
        assert abs(m0["rmse_v_steady"] - m1["rmse_v_steady"]) < 1e-4, f"{name}: zero residual changed rmse_v"
        print(f"2. {name}: zero residual == no residual (max torque diff {dtau:.2g} N*m, within GPU noise)")
        assert abs(m0["rmse_v_steady"] - ref) < 5e-4, f"{name}: rmse_v {m0['rmse_v_steady']:.4f} != Phase 1 {ref}"
        print(f"3. {name}: rmse_v_steady {m0['rmse_v_steady']:.4f} == Phase 1 reference {ref}")

    push = lambda t, x0, data, s: np.array([0, 0, 0.5, 0, 0, 0, 0, 0])
    m2, t2 = be.run_episode(mpc, model, be.SCENARIOS["vx_step"], residual_fn=push)
    m0, t0 = be.run_episode(mpc, model, be.SCENARIOS["vx_step"])
    n = min(len(t0["tau"]), len(t2["tau"]))
    dtau = float(np.max(np.abs(t0["tau"][:n] - t2["tau"][:n])))
    assert dtau > 10 * TAU_NOISE, f"non-zero residual had no effect (torque diff {dtau:.3g} N*m)"
    print(f"4. constant com_acc_z +0.5 m/s^2: max torque diff {dtau:.2f} N*m, rmse_v_steady "
          f"{m0['rmse_v_steady']:.4f} -> {m2['rmse_v_steady']:.4f}{' (fell)' if m2['fell'] else ''}: the hook is live")
    print("\nAll residual-hook checks passed.")


if __name__ == "__main__":
    main()
