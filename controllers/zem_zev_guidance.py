"""
Zero-Effort-Miss / Zero-Effort-Velocity (ZEM/ZEV) Optimal Guidance Law
and Analytical Minimum-Time Bisection Search Solver

Inertial Frame: ENU (East-North-Up)
Author: Sungcheol Ha (KAIST Flight Dynamics & GNC Research Project)
"""

import numpy as np
from typing import Tuple, Optional


class ZEMZEVGuidance:
    """
    Closed-form energy-optimal powered descent guidance law minimizing control effort:
        J = 0.5 * integral ||a_thrust(t)||^2 dt
    subject to:
        d/dt p = v
        d/dt v = a_thrust + g_W
    with boundary conditions:
        p(0) = p0, v(0) = v0, p(Tf) = pf, v(Tf) = vf.
    """

    def __init__(
        self,
        mass: float = 31.0,
        gravity: float = 9.80665,
        thrust_max: float = 600.0,
        tilt_max_deg: float = 35.0,
        t_eps: float = 0.20
    ):
        self.mass = float(mass)
        self.gravity = float(gravity)
        self.thrust_max = float(thrust_max)
        self.tilt_max_rad = np.deg2rad(float(tilt_max_deg))
        self.t_eps = float(t_eps)

        # In ENU coordinate frame: g_W = [0, 0, -g]^T
        self.g_vec = np.array([0.0, 0.0, -self.gravity], dtype=np.float64)

        # Maximum available thrust acceleration with tilt margin
        # a_max = (T_max / m) * cos(theta_max)
        self.a_max = (self.thrust_max / self.mass) * np.cos(self.tilt_max_rad)

    def compute_zem_zev(
        self,
        p: np.ndarray,
        v: np.ndarray,
        pf: np.ndarray,
        vf: np.ndarray,
        t_go: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates Zero-Effort-Miss (ZEM) and Zero-Effort-Velocity (ZEV) vectors.

        Args:
            p: Current vehicle position [x, y, z] in ENU [m] (3,)
            v: Current vehicle velocity [vx, vy, vz] in ENU [m/s] (3,)
            pf: Final desired landing position [m] (3,)
            vf: Final desired landing velocity [m/s] (3,)
            t_go: Time-to-go until touchdown [s]

        Returns:
            ZEM: Zero-effort miss vector [m] (3,)
            ZEV: Zero-effort velocity vector [m/s] (3,)
        """
        # Uncontrolled terminal state under gravity alone:
        # p_free(Tf) = p(t) + v(t) * t_go + 0.5 * g_W * t_go^2
        # v_free(Tf) = v(t) + g_W * t_go
        p_free = p + v * t_go + 0.5 * self.g_vec * (t_go ** 2)
        v_free = v + self.g_vec * t_go

        zem = pf - p_free
        zev = vf - v_free
        return zem, zev

    def compute_thrust_acceleration(
        self,
        p: np.ndarray,
        v: np.ndarray,
        pf: np.ndarray,
        vf: np.ndarray,
        t_go: float
    ) -> np.ndarray:
        """
        Computes the analytical optimal thrust acceleration vector:
            a_thrust = (6 / t_go^2) * ZEM - (2 / t_go) * ZEV

        Note:
            In ENU, total vehicle acceleration is:
                a_total = a_thrust + g_vec
            When ZEM = 0 and ZEV = 0 (equilibrium hover/cruise),
            the guidance law naturally commands a_thrust = [0, 0, +g]^T to counter gravity.
        """
        # Numerical safeguard against division by zero near touchdown
        t_go_safe = max(float(t_go), self.t_eps)

        zem, zev = self.compute_zem_zev(p, v, pf, vf, t_go_safe)
        a_thrust = (6.0 / (t_go_safe ** 2)) * zem - (2.0 / t_go_safe) * zev
        return a_thrust

    def estimate_peak_acceleration(
        self,
        p0: np.ndarray,
        v0: np.ndarray,
        pf: np.ndarray,
        vf: np.ndarray,
        t_go: float
    ) -> float:
        """
        O(1) analytical calculation of maximum ||a_thrust(tau)|| over tau in [0, t_go].

        Mathematical Theorem:
            Under optimal double-integrator control, a_thrust(tau) is affine in tau:
                a_thrust(tau) = c0 + c1 * tau
            The norm squared ||a_thrust(tau)||^2 is a strictly convex parabola opening upward.
            Therefore, its maximum on any compact interval [0, t_go] must occur at the boundaries:
                max ||a_thrust(tau)|| = max(||a_start||, ||a_end||)
            where:
                a_start = (6 / t_go^2) * ZEM - (2 / t_go) * ZEV
                a_end   = -(6 / t_go^2) * ZEM + (4 / t_go) * ZEV
        """
        t_go_safe = max(float(t_go), self.t_eps)
        zem, zev = self.compute_zem_zev(p0, v0, pf, vf, t_go_safe)

        a_start = (6.0 / (t_go_safe ** 2)) * zem - (2.0 / t_go_safe) * zev
        a_end = -(6.0 / (t_go_safe ** 2)) * zem + (4.0 / t_go_safe) * zev

        norm_start = float(np.linalg.norm(a_start))
        norm_end = float(np.linalg.norm(a_end))
        return max(norm_start, norm_end)

    def solve_optimal_tf(
        self,
        p0: np.ndarray,
        v0: np.ndarray,
        pf: np.ndarray,
        vf: np.ndarray,
        t_min: float = 2.0,
        t_max: float = 20.0,
        tol: float = 0.05,
        max_iter: int = 30
    ) -> float:
        """
        Solves for the minimum flight time Tf* subject to maximum acceleration bound:
            min Tf s.t. max_{t in [0, Tf]} ||a_thrust(t)|| <= a_max

        Uses fast bisection search leveraging the O(1) peak acceleration calculation.
        """
        # Ensure upper bound is feasible
        peak_at_max = self.estimate_peak_acceleration(p0, v0, pf, vf, t_max)
        if peak_at_max > self.a_max:
            # Expand t_max if initial bound is too tight
            t_max = t_max * 1.5

        t_low = float(t_min)
        t_high = float(t_max)

        for _ in range(max_iter):
            t_mid = 0.5 * (t_low + t_high)
            peak_accel = self.estimate_peak_acceleration(p0, v0, pf, vf, t_mid)

            if peak_accel <= self.a_max:
                t_high = t_mid  # Feasible, try smaller time
            else:
                t_low = t_mid   # Acceleration violation, increase time

            if (t_high - t_low) <= tol:
                break

        return t_high
