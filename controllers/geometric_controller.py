"""
Almost Globally Exponentially Stable SE(3) Geometric Tracking Controller
Lie Group SO(3) Attitude & Thrust Allocation for Quadrotor UAV

Reference:
  Lee, T., Leok, M., & McClamroch, N. H. (2010).
  "Geometric tracking control of a quadrotor UAV on SE(3)."
  49th IEEE Conference on Decision and Control (CDC), pp. 5420-5425.

Inertial Frame: ENU (East-North-Up)
Author: Sungcheol Ha (KAIST Flight Dynamics & GNC Research Project)
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional


class GeometricSE3Controller:
    """
    Nonlinear Geometric Controller on SE(3) avoiding Euler angle singularities (Gimbal Lock)
    and unwinding phenomena.
    """

    def __init__(
        self,
        mass: float = 31.0,
        J: Optional[np.ndarray] = None,
        k_R: Optional[np.ndarray] = None,
        k_Omega: Optional[np.ndarray] = None,
        thrust_min: float = 0.0,
        thrust_max: float = 600.0,
        torque_max: Optional[np.ndarray] = None,
        tilt_max_deg: float = 35.0
    ):
        self.mass = float(mass)
        self.J = np.diag([1.25, 1.25, 2.50]) if J is None else np.asarray(J, dtype=np.float64)
        self.J_inv = np.linalg.inv(self.J)

        # Attitude error gains (Diagonal)
        # Fast attitude convergence with proper damping for 31kg body
        self.k_R = np.array([250.0, 250.0, 70.0], dtype=np.float64) if k_R is None else np.asarray(k_R, dtype=np.float64)
        self.k_Omega = np.array([45.0, 45.0, 20.0], dtype=np.float64) if k_Omega is None else np.asarray(k_Omega, dtype=np.float64)

        self.thrust_min = float(thrust_min)
        self.thrust_max = float(thrust_max)
        self.torque_max = np.array([60.0, 60.0, 20.0], dtype=np.float64) if torque_max is None else np.asarray(torque_max, dtype=np.float64)
        self.tilt_max_rad = np.deg2rad(float(tilt_max_deg))

    def compute_desired_rotation(
        self,
        a_thrust_des: np.ndarray,
        yaw_des: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constructs desired orthonormal rotation matrix R_d in SO(3) from commanded thrust acceleration.

        Args:
            a_thrust_des: Commanded non-gravitational thrust acceleration vector [ax, ay, az] [m/s^2]
            yaw_des: Commanded heading angle [rad]

        Returns:
            R_d: Orthonormal rotation matrix in SO(3), Shape: (3, 3)
            b3_d: Desired body +Z axis direction vector in ENU, Shape: (3,)
        """
        a_cmd = a_thrust_des.copy()

        # 1. Enforce positive vertical lift to prevent inverted flight commands
        a_cmd[2] = max(a_cmd[2], 1.0)

        # 2. Geometric tilt angle clamping:
        # tan(theta) = ||a_xy|| / a_z <= tan(theta_max)
        a_xy_norm = np.linalg.norm(a_cmd[:2])
        max_a_xy = a_cmd[2] * np.tan(self.tilt_max_rad)

        if a_xy_norm > max_a_xy and a_xy_norm > 1e-6:
            scale = max_a_xy / a_xy_norm
            a_cmd[0] *= scale
            a_cmd[1] *= scale

        # 3. Desired body vertical axis (+Z_B in ENU): b3_d
        norm_a = np.linalg.norm(a_cmd)
        b3_d = a_cmd / norm_a if norm_a > 1e-6 else np.array([0.0, 0.0, 1.0], dtype=np.float64)

        # 4. Heading vector in horizontal plane: b1_c
        b1_c = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0], dtype=np.float64)

        # 5. Construct orthonormal basis vectors
        v_cross = np.cross(b3_d, b1_c)
        v_norm = np.linalg.norm(v_cross)

        # Handle collinear singularity (if b3_d is parallel to b1_c)
        if v_norm < 1e-4:
            b1_c_alt = np.array([-np.sin(yaw_des), np.cos(yaw_des), 0.0], dtype=np.float64)
            v_cross = np.cross(b3_d, b1_c_alt)
            v_norm = np.linalg.norm(v_cross)

        b2_d = v_cross / max(v_norm, 1e-6)
        b1_d = np.cross(b2_d, b3_d)
        b1_d = b1_d / np.linalg.norm(b1_d)

        # R_d = [b1_d, b2_d, b3_d]
        R_d = np.column_stack([b1_d, b2_d, b3_d])
        return R_d, b3_d

    def compute_control(
        self,
        R_curr: np.ndarray,
        omega_curr: np.ndarray,
        a_thrust_des: np.ndarray,
        yaw_des: float = 0.0
    ) -> Tuple[float, np.ndarray, Dict[str, Any]]:
        """
        Computes scalar collective thrust T and 3-axis control moment tau on SE(3).

        Args:
            R_curr: Current vehicle rotation matrix in SO(3), Shape: (3, 3)
            omega_curr: Current vehicle body angular velocity [p, q, r] [rad/s] (3,)
            a_thrust_des: Commanded thrust acceleration vector [m/s^2] (3,)
            yaw_des: Desired heading angle [rad]

        Returns:
            thrust: Collective total thrust [N] (float)
            torque: Body control torques [tau_x, tau_y, tau_z] [N*m] (3,)
            debug_info: Dictionary containing R_d, e_R, e_Omega, tilt_deg
        """
        # 1. Desired Rotation Matrix on SO(3)
        R_d, b3_d = self.compute_desired_rotation(a_thrust_des, yaw_des=yaw_des)

        # 2. Collective Thrust Allocation
        # Projected onto actual current body +Z axis (b3)
        b3_curr = R_curr[:, 2]
        thrust_raw = self.mass * float(np.dot(a_thrust_des, b3_curr))
        thrust = float(np.clip(thrust_raw, self.thrust_min, self.thrust_max))

        # 3. Attitude Error Vector on SO(3) (via Lie Algebra vee map)
        # e_R = 0.5 * (R_d^T * R - R^T * R_d)^\vee
        M = 0.5 * (R_d.T @ R_curr - R_curr.T @ R_d)
        e_R = np.array([M[2, 1], M[0, 2], M[1, 0]], dtype=np.float64)

        # 4. Angular Velocity Tracking Error
        # e_Omega = omega - R^T * R_d * omega_d (assuming omega_d = 0 for setpoint)
        e_Omega = omega_curr.copy()

        # 5. Control Torque with Gyroscopic Compensation
        # tau = -k_R * e_R - k_Omega * e_Omega + omega x (J * omega)
        tau_pd = -self.k_R * e_R - self.k_Omega * e_Omega
        tau_gyro = np.cross(omega_curr, self.J @ omega_curr)
        torque_raw = tau_pd + tau_gyro

        # Actuator torque saturation
        torque = np.clip(torque_raw, -self.torque_max, self.torque_max)

        # Current tilt angle
        tilt_deg = float(np.rad2deg(np.arccos(np.clip(b3_curr[2], -1.0, 1.0))))

        debug_info = {
            'R_d': R_d,
            'b3_d': b3_d,
            'e_R': e_R,
            'e_Omega': e_Omega,
            'tilt_deg': tilt_deg
        }

        return thrust, torque, debug_info
