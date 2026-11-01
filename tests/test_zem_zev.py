"""
Unit Tests for Phase 2: ZEM/ZEV Optimal Guidance and SE(3) Geometric Controller
Author: Sungcheol Ha (KAIST Flight Dynamics & GNC Research Project)
"""

import os
import sys
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from controllers.zem_zev_guidance import ZEMZEVGuidance
from controllers.geometric_controller import GeometricSE3Controller


def test_analytical_double_integrator_exact_match():
    """
    Scenario 1-A: Theoretical Closed-Form Equivalence.
    Verifies that the ZEM/ZEV analytical optimal acceleration solution
    drives the ideal double-integrator to the exact terminal state to machine precision (< 1e-12 m).
    """
    mass = 31.0
    gravity = 9.80665
    guidance = ZEMZEVGuidance(mass=mass, gravity=gravity, thrust_max=600.0, tilt_max_deg=35.0)

    p0 = np.array([-10.0, 0.0, 10.0], dtype=np.float64)
    v0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    pf = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    vf = np.array([0.0, 0.0, -0.15], dtype=np.float64)

    Tf = 3.5  # Fixed duration for exact step integration
    zem0, zev0 = guidance.compute_zem_zev(p0, v0, pf, vf, Tf)
    a_start = (6.0 / (Tf ** 2)) * zem0 - (2.0 / Tf) * zev0
    a_end = -(6.0 / (Tf ** 2)) * zem0 + (4.0 / Tf) * zev0

    n_steps = 3500
    dt = Tf / n_steps
    p = p0.copy()
    v = v0.copy()

    for step in range(n_steps):
        t_curr = step * dt
        a_th = a_start + (t_curr / Tf) * (a_end - a_start)
        a_th_mid = a_start + ((t_curr + 0.5 * dt) / Tf) * (a_end - a_start)
        a_th_end = a_start + ((t_curr + dt) / Tf) * (a_end - a_start)

        k1_v = a_th + guidance.g_vec
        k2_v = a_th_mid + guidance.g_vec
        k3_v = a_th_mid + guidance.g_vec
        k4_v = a_th_end + guidance.g_vec

        k1_p = v
        k2_p = v + 0.5 * dt * k1_v
        k3_p = v + 0.5 * dt * k2_v
        k4_p = v + dt * k3_v

        p += (dt / 6.0) * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p)
        v += (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

    pos_err = np.linalg.norm(p - pf)
    vel_err = np.linalg.norm(v - vf)

    assert pos_err < 1e-12, f"Theoretical double integrator pos_err mismatch: {pos_err:.4e}"
    assert vel_err < 1e-12, f"Theoretical double integrator vel_err mismatch: {vel_err:.4e}"
    print(f"[PASS] Scenario 1-A (Analytical Machine Precision): pos_err={pos_err:.2e} m, vel_err={vel_err:.2e} m/s")


def test_bisection_optimal_tf_search():
    """
    Scenario 1-B: Bisection Optimal Time-of-Flight Solver.
    Verifies that the bisection search converges to the minimum time
    satisfying maximum acceleration bound.
    """
    mass = 31.0
    gravity = 9.80665
    guidance = ZEMZEVGuidance(mass=mass, gravity=gravity, thrust_max=600.0, tilt_max_deg=35.0)

    p0 = np.array([-10.0, 0.0, 10.0], dtype=np.float64)
    v0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    pf = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    vf = np.array([0.0, 0.0, -0.15], dtype=np.float64)

    Tf = guidance.solve_optimal_tf(p0, v0, pf, vf, t_min=2.0, t_max=15.0, tol=0.01)
    peak_accel = guidance.estimate_peak_acceleration(p0, v0, pf, vf, Tf)

    assert 3.0 < Tf < 4.0, f"Optimal flight time out of expected physical range: Tf = {Tf:.2f} s"
    assert peak_accel <= guidance.a_max + 0.05, f"Peak acceleration {peak_accel:.2f} exceeds a_max {guidance.a_max:.2f}"
    print(f"[PASS] Scenario 1-B (Optimal Tf Bisection): Solved Tf* = {Tf:.3f} s (Peak Accel: {peak_accel:.2f} <= {guidance.a_max:.2f} m/s^2)")


def test_se3_attitude_tracking_lyapunov():
    """
    Scenario 2: SE(3) Attitude Tracking & Lyapunov Stability.
    Verifies that the geometric controller recovers from a large initial tilt (30 deg pitch, 20 deg roll)
    and drives the attitude error metric Psi = 0.5 * tr(I - Rd^T * R) to zero exponentially.
    """
    mass = 31.0
    J = np.diag([1.25, 1.25, 2.50])
    J_inv = np.linalg.inv(J)
    ctrl = GeometricSE3Controller(mass=mass, J=J)

    # Initial attitude: Roll = 20 deg, Pitch = 30 deg, Yaw = 10 deg
    roll0 = np.deg2rad(20.0)
    pitch0 = np.deg2rad(30.0)
    yaw0 = np.deg2rad(10.0)

    # Rotation matrix ZYX
    Rz = np.array([[np.cos(yaw0), -np.sin(yaw0), 0], [np.sin(yaw0), np.cos(yaw0), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(pitch0), 0, np.sin(pitch0)], [0, 1, 0], [-np.sin(pitch0), 0, np.cos(pitch0)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(roll0), -np.sin(roll0)], [0, np.sin(roll0), np.cos(roll0)]])
    R_curr = Rz @ Ry @ Rx
    omega_curr = np.zeros(3, dtype=np.float64)

    # Desired command: Pure vertical hover (az = +9.80665 m/s^2), Yaw = 0 -> Rd = Identity
    a_thrust_des = np.array([0.0, 0.0, 9.80665], dtype=np.float64)

    dt = 0.002
    sim_time = 1.0  # 1.0 second response
    n_steps = int(sim_time / dt)

    for _ in range(n_steps):
        thrust, torque, info = ctrl.compute_control(R_curr, omega_curr, a_thrust_des, yaw_des=0.0)

        # Rotational Euler dynamics: J * omega_dot = tau - omega x (J * omega)
        def rot_deriv(R, w):
            tau_cmd = ctrl.compute_control(R, w, a_thrust_des, yaw_des=0.0)[1]
            w_dot = J_inv @ (tau_cmd - np.cross(w, J @ w))
            wx = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
            R_dot = R @ wx
            return R_dot, w_dot

        k1_R, k1_w = rot_deriv(R_curr, omega_curr)
        k2_R, k2_w = rot_deriv(R_curr + 0.5 * dt * k1_R, omega_curr + 0.5 * dt * k1_w)
        k3_R, k3_w = rot_deriv(R_curr + 0.5 * dt * k2_R, omega_curr + 0.5 * dt * k2_w)
        k4_R, k4_w = rot_deriv(R_curr + dt * k3_R, omega_curr + dt * k3_w)

        R_curr += (dt / 6.0) * (k1_R + 2.0 * k2_R + 2.0 * k3_R + k4_R)
        omega_curr += (dt / 6.0) * (k1_w + 2.0 * k2_w + 2.0 * k3_w + k4_w)

        # Orthonormalize R_curr via SVD projection onto SO(3)
        U, _, Vt = np.linalg.svd(R_curr)
        R_curr = U @ Vt
        if np.linalg.det(R_curr) < 0:
            U[:, -1] *= -1
            R_curr = U @ Vt

    # Compute Lyapunov attitude error metric Psi = 0.5 * tr(I - Rd^T * R)
    R_d = np.eye(3)
    Psi = 0.5 * np.trace(np.eye(3) - R_d.T @ R_curr)
    omega_norm = np.linalg.norm(omega_curr)

    assert Psi < 1e-3, f"SE(3) attitude did not converge: Psi = {Psi:.6e}"
    assert omega_norm < 1e-2, f"Angular rate did not damp out: ||omega|| = {omega_norm:.6e} rad/s"
    print(f"[PASS] Scenario 2 (SE(3) Attitude Tracking): Psi={Psi:.2e}, ||omega||={omega_norm:.2e} rad/s (Smooth exponential convergence)")


def test_singularity_avoidance_terminal_t_go():
    """
    Scenario 3: Singularity Avoidance and Acceleration Boundedness.
    Verifies that as t_go -> 0 and below t_eps, compute_thrust_acceleration does not blow up
    or produce NaN / Inf values.
    """
    guidance = ZEMZEVGuidance(mass=31.0, gravity=9.80665, thrust_max=600.0, tilt_max_deg=35.0, t_eps=0.20)

    p = np.array([0.02, 0.01, 0.05])
    v = np.array([0.05, 0.00, -0.15])
    pf = np.array([0.0, 0.0, 0.0])
    vf = np.array([0.0, 0.0, -0.15])

    for t_go in [1.0, 0.5, 0.2, 0.1, 0.01, 0.0, -0.1]:
        a_th = guidance.compute_thrust_acceleration(p, v, pf, vf, t_go)
        assert not np.any(np.isnan(a_th)), f"NaN detected at t_go = {t_go}"
        assert not np.any(np.isinf(a_th)), f"Inf detected at t_go = {t_go}"
        assert np.linalg.norm(a_th) < 100.0, f"Unbounded acceleration at t_go = {t_go}: ||a_th|| = {np.linalg.norm(a_th)}"

    print("[PASS] Scenario 3 (Singularity Avoidance): Guidance acceleration is strictly bounded as t_go -> 0.")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(">>> Running Phase 2 ZEM/ZEV & SE(3) Unit Tests")
    print("=" * 60)
    test_analytical_double_integrator_exact_match()
    test_bisection_optimal_tf_search()
    test_se3_attitude_tracking_lyapunov()
    test_singularity_avoidance_terminal_t_go()
    print("=" * 60)
    print("ALL PHASE 2 UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 60 + "\n")
