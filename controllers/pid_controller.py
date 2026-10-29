"""
3-Phase Decoupled Sequential PID Controller for 6-DOF Multirotor
Stable and Robust Waypoint Navigation & Precision Soft Landing

Author: Sungcheol Ha (KAIST Flight Dynamics & GNC Research Project)
"""

import numpy as np
from typing import Dict, Any, Optional
from controllers.base_controller import BaseController


class SegmentedPIDController(BaseController):
    """
    Robust 3-Phase Waypoint Guidance & Cascaded PID Controller:
      - Phase 1: Descend vertically at start point: (-10, 0, 10) -> (-10, 0, 5) m
      - Phase 2: Horizontal cruise at constant altitude: (-10, 0, 5) -> (0, 0, 5) m
      - Phase 3: Vertical precision soft touchdown with flare: (0, 0, 5) -> (0, 0, 0) m
    """

    def __init__(self, drone_cfg: Optional[dict] = None, mission_cfg: Optional[dict] = None):
        # 1. Drone Physical Parameters
        self.mass: float = 31.0
        self.gravity: float = 9.80665
        self.thrust_max: float = 600.0
        self.thrust_min: float = 0.0
        self.tilt_max_rad: float = np.deg2rad(25.0)  # Gentle tilt (<= 25 deg) to prevent aggressive pitch coupling
        self.torque_max: np.ndarray = np.array([60.0, 60.0, 20.0], dtype=np.float64)

        if drone_cfg is not None:
            self.mass = float(drone_cfg.get('mass', 31.0))
            self.gravity = float(drone_cfg.get('gravity', 9.80665))
            geom = drone_cfg.get('geometry', {})
            self.thrust_max = float(geom.get('thrust_max', 600.0))
            self.thrust_min = float(geom.get('thrust_min', 0.0))
            torq_cfg = geom.get('torque_max', {})
            self.torque_max = np.array([
                float(torq_cfg.get('roll', 60.0)),
                float(torq_cfg.get('pitch', 60.0)),
                float(torq_cfg.get('yaw', 20.0))
            ], dtype=np.float64)

        # 2. Position Loop PID Gains
        self.Kp_pos: np.ndarray = np.array([2.5, 1.5, 1.5], dtype=np.float64)
        self.Ki_pos: np.ndarray = np.array([0.1, 0.1, 0.1], dtype=np.float64)
        self.Kd_pos: np.ndarray = np.array([2.0, 1.5, 2.5], dtype=np.float64)  # Proper damping for 31kg body

        # 3. Attitude Loop PD Gains (Fast attitude tracking)
        self.Kp_att: np.ndarray = np.array([180.0, 180.0, 50.0], dtype=np.float64)
        self.Kd_att: np.ndarray = np.array([35.0, 35.0, 15.0], dtype=np.float64)

        # 4. Safe Speed Limits for 31kg Quadrotor
        self.v_max_x: float = 3.6    # Horizontal cruise speed limit [m/s]
        self.v_max_z: float = 3.2    # Vertical descent speed limit [m/s]

        # 5. Waypoints (Decoupled 3-Phase Progression)
        self.wp1 = np.array([-10.0, 0.0, 5.0], dtype=np.float64)
        self.wp2 = np.array([0.0, 0.0, 5.0], dtype=np.float64)
        self.wp3 = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # Internal States
        self.phase: int = 1
        self.intErr: np.ndarray = np.zeros(3, dtype=np.float64)

    def reset(self) -> None:
        """Resets controller internal states."""
        self.phase = 1
        self.intErr.fill(0.0)

    @staticmethod
    def _quat_to_euler(q: np.ndarray) -> np.ndarray:
        qw, qx, qy, qz = q
        # Roll
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch
        sinp = 2.0 * (qw * qy - qz * qx)
        sinp = np.clip(sinp, -1.0 + 1e-7, 1.0 - 1e-7)
        pitch = np.arcsin(sinp)

        # Yaw
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw], dtype=np.float64)

    def compute_control(
        self,
        state_dict: Dict[str, np.ndarray],
        t: float,
        dt: float
    ) -> Dict[str, Any]:
        """
        Computes 4-DOF control [T, tau_x, tau_y, tau_z] ensuring stable waypoint progression.
        """
        pos = state_dict['p'].copy()
        vel = state_dict['v'].copy()
        q = state_dict['q']
        omega = state_dict['omega']

        # 1. Sequential FSM Phase Transition
        if self.phase == 1:
            p_target = self.wp1
            # Transition when altitude reaches 5.0m and vertical speed stabilized
            if pos[2] <= 5.20 and abs(vel[2]) < 0.8:
                self.phase = 2
                self.intErr.fill(0.0)
        elif self.phase == 2:
            p_target = self.wp2
            dist_xy = np.linalg.norm(pos[:2])
            # Transition when arrived over target pad and horizontally stabilized
            if dist_xy <= 0.25 and abs(vel[0]) < 0.6:
                self.phase = 3
                self.intErr.fill(0.0)
        else:
            p_target = self.wp3

        # 2. Position and Velocity Error Calculation
        e_p = p_target - pos
        e_p[0] = np.clip(e_p[0], -self.v_max_x, self.v_max_x)
        e_p[1] = np.clip(e_p[1], -self.v_max_x, self.v_max_x)

        if self.phase == 3:
            # Phase 3: Vertical Landing with Flare Sink-rate Regulation (Soft Landing v <= 0.20 m/s)
            vz_des = -float(np.clip(0.12 + 0.50 * pos[2], 0.12, 2.2))
            e_vz = vz_des - vel[2]
            az_cmd = 3.5 * e_vz + 0.2 * self.intErr[2]
            self.intErr[2] = np.clip(self.intErr[2] + e_vz * dt, -2.0, 2.0)

            # Lateral stabilization during vertical descent
            ax_cmd = self.Kp_pos[0] * e_p[0] + self.Ki_pos[0] * self.intErr[0] - self.Kd_pos[0] * vel[0]
            ay_cmd = self.Kp_pos[1] * e_p[1] + self.Ki_pos[1] * self.intErr[1] - self.Kd_pos[1] * vel[1]
            self.intErr[0] = np.clip(self.intErr[0] + e_p[0] * dt, -2.0, 2.0)
            self.intErr[1] = np.clip(self.intErr[1] + e_p[1] * dt, -2.0, 2.0)
            a_cmd = np.array([ax_cmd, ay_cmd, az_cmd], dtype=np.float64)
        else:
            e_p[2] = np.clip(e_p[2], -self.v_max_z, self.v_max_z)
            self.intErr += e_p * dt
            self.intErr = np.clip(self.intErr, -3.0, 3.0)
            a_cmd = self.Kp_pos * e_p + self.Ki_pos * self.intErr - self.Kd_pos * vel

        # 3. Attitude and Thrust Mapping
        az_tot = max(a_cmd[2] + self.gravity, 1.0)

        theta_d = np.clip(np.arctan(a_cmd[0] / az_tot), -self.tilt_max_rad, self.tilt_max_rad)
        phi_d = np.clip(np.arctan(-a_cmd[1] / az_tot * np.cos(theta_d)), -self.tilt_max_rad, self.tilt_max_rad)
        psi_d = 0.0

        euler = self._quat_to_euler(q)
        roll, pitch, yaw = euler[0], euler[1], euler[2]

        cos_roll = max(np.cos(roll), 0.2)
        cos_pitch = max(np.cos(pitch), 0.2)
        thrust = (self.mass * az_tot) / (cos_roll * cos_pitch)
        thrust = np.clip(thrust, self.thrust_min, self.thrust_max)

        # 4. Attitude Loop (PD Torque Control)
        e_roll = phi_d - roll
        e_pitch = theta_d - pitch
        e_yaw = psi_d - yaw
        e_att = np.array([e_roll, e_pitch, e_yaw], dtype=np.float64)

        tau = self.Kp_att * e_att - self.Kd_att * omega
        tau = np.clip(tau, -self.torque_max, self.torque_max)

        u_raw = np.array([thrust, tau[0], tau[1], tau[2]], dtype=np.float64)

        return {
            'u_raw': u_raw,
            'thrust': thrust,
            'torque': tau,
            'phase': self.phase,
            'target_pos': p_target,
            'a_cmd': a_cmd,
            'att_desired': np.array([phi_d, theta_d, psi_d])
        }
