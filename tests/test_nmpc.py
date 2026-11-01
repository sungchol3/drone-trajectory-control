"""
Unit Tests for Phase 3: CasADi Nonlinear Model Predictive Control (NMPC)
High-Fidelity 6-DOF Multirotor Precision Trajectory and Landing Verification

Test Scenarios:
  - Scenario 1: OCP Formulation & Cold-Start Feasibility
  - Scenario 2: Hard Inequality Constraint Satisfaction (Ground, Tilt, Thrust)
  - Scenario 3: Warm-Start Execution Time Benchmark (<25ms for real-time RHC)
  - Scenario 4: Terminal Soft-Landing Touchdown Sink-rate Compliance (vz >= -0.20 m/s)
  - Scenario 5: NMPCController Conformance to BaseController Interface

Author: Sungcheol Ha (KAIST Flight Dynamics & GNC Research Project)
"""

import sys
import os
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from controllers.nmpc_casadi_model import CasADiNMPCModel, NMPCConfig
from controllers.nmpc_controller import NMPCController


def test_scenario_1_ocp_cold_start():
    """Verifies CasADi OCP symbolic compilation and cold-start convergence."""
    print("\n--- Running Scenario 1: OCP Cold-Start Feasibility ---")
    cfg = NMPCConfig(N=20, dt_mpc=0.05, print_level=0)
    model = CasADiNMPCModel(cfg)

    x_init = np.array([-10.0, 0.0, 10.0, 0.0, 0.0, 0.0], dtype=np.float64)
    x_target = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.15], dtype=np.float64)

    success, u0, w_opt, info = model.solve(x_init, x_target)

    print(f"  Solver Status:       {info['status']}")
    print(f"  Cold-start Time:     {info['solve_time_ms']:.2f} ms")
    print(f"  Initial Control u0:  {u0} m/s^2")

    assert success, f"OCP cold start failed with status: {info['status']}"
    assert u0[2] > 0.0, f"Vertical thrust acceleration must be positive: u0[2]={u0[2]}"
    print("  [PASS] Scenario 1: OCP cold start converged successfully.")


def test_scenario_2_hard_constraints_satisfaction():
    """Verifies that all hard state and actuator inequality constraints are strictly satisfied."""
    print("\n--- Running Scenario 2: Hard Constraints Satisfaction ---")
    cfg = NMPCConfig(N=25, dt_mpc=0.04, print_level=0, z_min=0.04, tilt_max_deg=35.0)
    model = CasADiNMPCModel(cfg)

    x_init = np.array([-10.0, 0.0, 10.0, 2.0, 0.0, -1.0], dtype=np.float64)
    x_target = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.15], dtype=np.float64)

    success, u0, w_opt, info = model.solve(x_init, x_target)
    assert success, f"OCP solve failed: {info['status']}"

    X_opt = info['X_opt']
    U_opt = info['U_opt']

    # 1. Ground clearance: z_k >= z_min (with 1e-4 numerical tolerance)
    min_z = float(np.min(X_opt[:, 2]))
    print(f"  Minimum Altitude across Horizon: {min_z:.4f} m (Bound: >= {cfg.z_min:.4f} m)")
    assert min_z >= cfg.z_min - 1e-4, f"Ground penetration violation: min_z={min_z} < {cfg.z_min}"

    # 2. Maximum thrust saturation: ||u_k|| <= T_max / mass
    max_accel_bound = cfg.thrust_max / cfg.mass
    thrust_mags = np.linalg.norm(U_opt, axis=1)
    max_thrust_mag = float(np.max(thrust_mags))
    print(f"  Peak Control Acceleration:       {max_thrust_mag:.4f} m/s^2 (Bound: <= {max_accel_bound:.4f} m/s^2)")
    assert max_thrust_mag <= max_accel_bound + 1e-3, f"Thrust limit exceeded: {max_thrust_mag} > {max_accel_bound}"

    # 3. Tilt angle cone: ||u_xy|| / u_z <= tan(theta_max)
    tan_tilt_max = np.tan(np.deg2rad(cfg.tilt_max_deg))
    xy_norms = np.linalg.norm(U_opt[:, :2], axis=1)
    tan_tilts = xy_norms / np.maximum(U_opt[:, 2], 1e-3)
    max_tilt_deg = float(np.rad2deg(np.arctan(np.max(tan_tilts))))
    print(f"  Peak Predicted Tilt Angle:       {max_tilt_deg:.2f}° (Bound: <= {cfg.tilt_max_deg:.2f}°)")
    assert max_tilt_deg <= cfg.tilt_max_deg + 0.1, f"Tilt angle constraint violated: {max_tilt_deg}° > {cfg.tilt_max_deg}°"

    print("  [PASS] Scenario 2: All hard constraints strictly satisfied.")


def test_scenario_3_warm_start_benchmark():
    """Benchmarks RHC warm-start solve times across successive steps."""
    print("\n--- Running Scenario 3: Warm-Start Execution Time Benchmark ---")
    cfg = NMPCConfig(N=25, dt_mpc=0.04, print_level=0)
    model = CasADiNMPCModel(cfg)

    x_curr = np.array([-10.0, 0.0, 10.0, 0.0, 0.0, 0.0], dtype=np.float64)
    x_target = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.15], dtype=np.float64)

    # Initial cold solve
    success, u0, w_opt, info = model.solve(x_curr, x_target)
    assert success

    solve_times = []
    # Simulate 10 receding horizon steps
    for step in range(10):
        # Shift solution forward
        w_guess = model.shift_solution(w_opt)
        # Advance state along predicted trajectory
        X_opt = info['X_opt']
        x_curr = X_opt[1].copy()

        t_start = time.perf_counter()
        success, u0, w_opt, info = model.solve(x_curr, x_target, w_guess=w_guess)
        t_ms = (time.perf_counter() - t_start) * 1000.0
        assert success
        solve_times.append(t_ms)

    mean_time = float(np.mean(solve_times))
    max_time = float(np.max(solve_times))
    print(f"  Warm-start Mean Solve Time: {mean_time:.2f} ms")
    print(f"  Warm-start Max Solve Time:  {max_time:.2f} ms")
    print(f"  Effective RHC Rate:         {1000.0 / mean_time:.1f} Hz")

    assert mean_time < 25.0, f"Mean warm-start solve time too high for real-time: {mean_time:.2f} ms >= 25 ms"
    print("  [PASS] Scenario 3: Warm-start real-time efficiency verified.")


def test_scenario_4_terminal_soft_touchdown():
    """Verifies that terminal velocity strictly respects soft touchdown sink rate (vz >= -0.20 m/s)."""
    print("\n--- Running Scenario 4: Terminal Soft-Landing Sink Rate ---")
    cfg = NMPCConfig(N=25, dt_mpc=0.04, print_level=0, v_touchdown=0.20)
    model = CasADiNMPCModel(cfg)

    # Approaching touchdown from low altitude
    x_init = np.array([-0.5, 0.0, 1.2, 0.2, 0.0, -0.8], dtype=np.float64)
    x_target = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -0.15], dtype=np.float64)

    success, u0, w_opt, info = model.solve(x_init, x_target)
    assert success

    X_opt = info['X_opt']
    terminal_vz = float(X_opt[-1, 5])
    print(f"  Terminal Vertical Velocity (vz,N): {terminal_vz:.4f} m/s (Bound: >= {-cfg.v_touchdown:.4f} m/s)")

    assert terminal_vz >= -cfg.v_touchdown - 1e-4, f"Terminal sink rate too fast for soft landing: {terminal_vz} < {-cfg.v_touchdown}"
    assert terminal_vz <= 0.05, f"Terminal velocity should not be climbing: {terminal_vz} > 0.05"
    print("  [PASS] Scenario 4: Terminal soft touchdown constraint strictly satisfied.")


def test_scenario_5_nmpc_controller_interface():
    """Verifies NMPCController conforms to BaseController and executes smoothly."""
    print("\n--- Running Scenario 5: NMPCController Interface Conformance ---")
    controller = NMPCController()
    controller.reset()

    # Mock quadrotor state
    state = {
        'p': np.array([-10.0, 0.0, 10.0], dtype=np.float64),
        'v': np.array([0.0, 0.0, 0.0], dtype=np.float64),
        'q': np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        'R': np.eye(3, dtype=np.float64),
        'omega': np.array([0.0, 0.0, 0.0], dtype=np.float64),
        'time': 0.0
    }

    out = controller.compute_control(state, t=0.0, dt=0.005)

    assert 'thrust' in out
    assert 'torque' in out
    assert 'u_raw' in out
    assert 'info' in out
    assert isinstance(out['thrust'], float)
    assert out['torque'].shape == (3,)
    assert 0.0 <= out['thrust'] <= 600.0

    print(f"  Commanded Thrust: {out['thrust']:.2f} N")
    print(f"  Commanded Torque: {out['torque']} N*m")
    print("  [PASS] Scenario 5: NMPCController BaseController interface verified.")


if __name__ == '__main__':
    print("============================================================")
    print(">>> Running Phase 3 NMPC Unit Tests")
    print("============================================================")
    test_scenario_1_ocp_cold_start()
    test_scenario_2_hard_constraints_satisfaction()
    test_scenario_3_warm_start_benchmark()
    test_scenario_4_terminal_soft_touchdown()
    test_scenario_5_nmpc_controller_interface()
    print("============================================================")
    print("ALL PHASE 3 NMPC UNIT TESTS PASSED SUCCESSFULLY!")
    print("============================================================")
