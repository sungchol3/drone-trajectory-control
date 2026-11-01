"""
Unified ZEM/ZEV Optimal Guidance & SE(3) Geometric Flight Controller
Inertial Frame: ENU (East-North-Up)

Author: Sungcheol Ha (KAIST Flight Dynamics & GNC Research Project)
"""

import numpy as np
from typing import Dict, Any, Optional
from controllers.base_controller import BaseController
from controllers.zem_zev_guidance import ZEMZEVGuidance
from controllers.geometric_controller import GeometricSE3Controller


class ZEMZEVController(BaseController):
    """
    Unified Hierarchical Optimal Guidance and SE(3) Tracking Controller:
      - High-level: ZEM/ZEV closed-form analytical acceleration guidance with bisection optimal time search
      - Low-level: SE(3) geometric tracking attitude control
      - Terminal: Soft touchdown ground flare regulation
    """

    def __init__(
        self,
        drone_cfg: Optional[dict] = None,
        mission_cfg: Optional[dict] = None
    ):
        # 1. Physical parameters
        self.mass: float = 31.0
        self.gravity: float = 9.80665
        self.thrust_max: float = 600.0
        self.thrust_min: float = 0.0
        self.tilt_max_deg: float = 35.0
        self.torque_max: np.ndarray = np.array([60.0, 60.0, 20.0], dtype=np.float64)
        self.J: np.ndarray = np.diag([1.25, 1.25, 2.50])

        if drone_cfg is not None:
            self.mass = float(drone_cfg.get('mass', 31.0))
            self.gravity = float(drone_cfg.get('gravity', 9.80665))
            geom = drone_cfg.get('geometry', {})
            self.thrust_max = float(geom.get('thrust_max', 600.0))
            self.thrust_min = float(geom.get('thrust_min', 0.0))
            self.tilt_max_deg = float(geom.get('tilt_max_deg', 35.0))
            torq_cfg = geom.get('torque_max', {})
            self.torque_max = np.array([
                float(torq_cfg.get('roll', 60.0)),
                float(torq_cfg.get('pitch', 60.0)),
                float(torq_cfg.get('yaw', 20.0))
            ], dtype=np.float64)
            iner = drone_cfg.get('inertia', {})
            self.J = np.diag([
                float(iner.get('Ixx', 1.25)),
                float(iner.get('Iyy', 1.25)),
                float(iner.get('Izz', 2.50))
            ])

        # 2. Target waypoints
        self.start_pos = np.array([-10.0, 0.0, 10.0], dtype=np.float64)
        self.start_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.target_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        # Touchdown soft landing velocity (-0.15 m/s sink rate)
        self.target_vel = np.array([0.0, 0.0, -0.15], dtype=np.float64)

        if mission_cfg is not None:
            wp = mission_cfg.get('waypoints', {})
            self.start_pos = np.array(wp.get('start_pos', [-10.0, 0.0, 10.0]), dtype=np.float64)
            self.start_vel = np.array(wp.get('start_vel', [0.0, 0.0, 0.0]), dtype=np.float64)
            self.target_pos = np.array(wp.get('target_pos', [0.0, 0.0, 0.0]), dtype=np.float64)

        # 3. Instantiate Submodules
        self.guidance = ZEMZEVGuidance(
            mass=self.mass,
            gravity=self.gravity,
            thrust_max=self.thrust_max,
            tilt_max_deg=self.tilt_max_deg,
            t_eps=0.20
        )

        self.att_ctrl = GeometricSE3Controller(
            mass=self.mass,
            J=self.J,
            thrust_min=self.thrust_min,
            thrust_max=self.thrust_max,
            torque_max=self.torque_max,
            tilt_max_deg=self.tilt_max_deg
        )

        # 4. Controller States
        self.phase: int = 1  # 1: ZEM/ZEV Guidance, 2: Terminal Flare Landing
        self.Tf_optimal: float = 6.0
        self.is_tf_solved: bool = False

    def reset(self) -> None:
        """Resets controller internal states and solves optimal flight time."""
        self.phase = 1
        self.is_tf_solved = False

    def solve_mission_time(self, init_pos: np.ndarray, init_vel: np.ndarray) -> float:
        """
        Computes optimal minimum flight time via Bisection search,
        adding a small margin (0.45s) to account for 6-DOF physical attitude tracking lag.
        """
        raw_tf = self.guidance.solve_optimal_tf(
            p0=init_pos,
            v0=init_vel,
            pf=self.target_pos,
            vf=self.target_vel,
            t_min=2.0,
            t_max=20.0,
            tol=0.05
        )
        self.Tf_optimal = raw_tf + 0.45
        self.is_tf_solved = True
        return self.Tf_optimal

    def compute_control(
        self,
        state_dict: Dict[str, np.ndarray],
        t: float,
        dt: float
    ) -> Dict[str, Any]:
        """
        Computes 4-DOF control commands [T, tau_x, tau_y, tau_z] conforming to BaseController.
        """
        pos = state_dict['p'].copy()
        vel = state_dict['v'].copy()
        R_curr = state_dict['R'].copy()
        omega_curr = state_dict['omega'].copy()

        # Solve optimal flight time on first control step
        if not self.is_tf_solved:
            self.solve_mission_time(pos, vel)

        t_go = max(self.Tf_optimal - t, 0.0)

        # 1. State Machine & Terminal Handover
        # Transition to terminal soft flare mode when close to touchdown to avoid 1/t_go singularities
        # and ensure smooth sink-rate deceleration (< 0.20 m/s)
        if self.phase == 1 and (t_go <= 0.50 or pos[2] <= 1.20):
            self.phase = 2

        # 2. Guidance Acceleration Computation
        if self.phase == 1:
            # ZEM/ZEV optimal closed-form acceleration
            t_go_safe = max(t_go, 0.20)
            a_thrust = self.guidance.compute_thrust_acceleration(
                p=pos,
                v=vel,
                pf=self.target_pos,
                vf=self.target_vel,
                t_go=t_go_safe
            )
        else:
            # Terminal Precision Flare Landing (z <= 1.20m)
            # Lateral position hold over pad
            e_xy = self.target_pos[:2] - pos[:2]
            ax = 2.5 * e_xy[0] - 2.8 * vel[0]
            ay = 2.5 * e_xy[1] - 2.8 * vel[1]

            # Soft landing sink rate regulation: smooth attenuation to -0.15 m/s
            vz_des = -float(np.clip(0.12 + 0.45 * pos[2], 0.12, 1.2))
            az = 4.0 * (vz_des - vel[2]) + self.gravity
            a_thrust = np.array([ax, ay, az], dtype=np.float64)

        # 3. SE(3) Geometric Attitude and Thrust Control
        thrust, torque, debug_info = self.att_ctrl.compute_control(
            R_curr=R_curr,
            omega_curr=omega_curr,
            a_thrust_des=a_thrust,
            yaw_des=0.0
        )

        u_raw = np.array([thrust, torque[0], torque[1], torque[2]], dtype=np.float64)

        return {
            'u_raw': u_raw,
            'thrust': thrust,
            'torque': torque,
            'phase': self.phase,
            'target_pos': self.target_pos,
            'a_cmd': a_thrust,
            't_go': t_go,
            'Tf_optimal': self.Tf_optimal,
            'info': debug_info
        }
