"""
Unified Nonlinear Model Predictive Control (NMPC) & SE(3) Flight Controller
High-Fidelity 6-DOF Multirotor Autonomous Trajectory Tracking and Soft Landing

Architecture:
  - High-level: CasADi Multiple Shooting NMPC solver (25~50 Hz)
      * Explicit hard ground constraint (z >= 0.04m)
      * Tilt cone inequality constraint (<= 35 deg)
      * Maximum collective thrust saturation (0 <= T <= 600N)
      * Terminal soft touchdown sink-rate regulation (vz >= -0.20 m/s)
  - Low-level: Lie Group SE(3) geometric attitude and torque tracking (200 Hz)
  - Execution: Multi-rate Receding Horizon Control with warm-starting (<15ms)
  - Safety: Graceful fallback to previous prediction or ZEM/ZEV guidance

Author: Sungcheol Ha (KAIST Flight Dynamics & GNC Research Project)
"""

import numpy as np
from typing import Dict, Any, Optional
from controllers.base_controller import BaseController
from controllers.nmpc_casadi_model import CasADiNMPCModel, NMPCConfig
from controllers.geometric_controller import GeometricSE3Controller
from controllers.zem_zev_guidance import ZEMZEVGuidance


class NMPCController(BaseController):
    """
    Unified Hierarchical NMPC and SE(3) Geometric Flight Controller conforming to BaseController.
    """

    def __init__(
        self,
        drone_cfg: Optional[dict] = None,
        mission_cfg: Optional[dict] = None
    ):
        # 1. Physical Parameters
        self.mass: float = 31.0
        self.gravity: float = 9.80665
        self.thrust_max: float = 600.0
        self.thrust_min: float = 0.0
        self.tilt_max_deg: float = 35.0
        self.torque_max: np.ndarray = np.array([60.0, 60.0, 20.0], dtype=np.float64)
        self.J: np.ndarray = np.diag([1.25, 1.25, 2.50])
        self.drag_coeffs: np.ndarray = np.array([0.1, 0.1, 0.1], dtype=np.float64)

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
            drg = drone_cfg.get('drag', {})
            self.drag_coeffs = np.array([
                float(drg.get('dx', 0.1)),
                float(drg.get('dy', 0.1)),
                float(drg.get('dz', 0.1))
            ], dtype=np.float64)

        # 2. Mission Waypoints & Timing
        self.start_pos = np.array([-10.0, 0.0, 10.0], dtype=np.float64)
        self.start_vel = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.target_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.target_vel = np.array([0.0, 0.0, -0.15], dtype=np.float64)
        self.dt_mpc: float = 0.04  # 25 Hz NMPC execution
        self.dt_sim: float = 0.005 # 200 Hz inner loop

        if mission_cfg is not None:
            wp = mission_cfg.get('waypoints', {})
            self.start_pos = np.array(wp.get('start_pos', [-10.0, 0.0, 10.0]), dtype=np.float64)
            self.start_vel = np.array(wp.get('start_vel', [0.0, 0.0, 0.0]), dtype=np.float64)
            self.target_pos = np.array(wp.get('target_pos', [0.0, 0.0, 0.0]), dtype=np.float64)
            tm = mission_cfg.get('timing', {})
            self.dt_sim = float(tm.get('dt_sim', 0.005))

        # 3. Instantiate CasADi NMPC Model
        self.nmpc_cfg = NMPCConfig(
            N=25,
            dt_mpc=self.dt_mpc,
            mass=self.mass,
            gravity=self.gravity,
            thrust_min=self.thrust_min,
            thrust_max=self.thrust_max,
            tilt_max_deg=self.tilt_max_deg,
            drag_coeffs=self.drag_coeffs,
            z_min=0.04,
            v_touchdown=0.20,
            print_level=0
        )
        self.nmpc_model = CasADiNMPCModel(self.nmpc_cfg)

        # 4. Instantiate Inner SE(3) Geometric Controller
        self.att_ctrl = GeometricSE3Controller(
            mass=self.mass,
            J=self.J,
            thrust_min=self.thrust_min,
            thrust_max=self.thrust_max,
            torque_max=self.torque_max,
            tilt_max_deg=self.tilt_max_deg
        )

        # 5. Backup ZEM/ZEV Guidance for emergency fallback
        self.fallback_guidance = ZEMZEVGuidance(
            mass=self.mass,
            gravity=self.gravity,
            thrust_max=self.thrust_max,
            tilt_max_deg=self.tilt_max_deg
        )

        # 6. Controller State Buffers & Solver Reliability Metrics
        self.w_opt_prev: Optional[np.ndarray] = None
        self.last_mpc_time: float = -1.0
        self.current_a_thrust_cmd: np.ndarray = np.array([0.0, 0.0, self.gravity], dtype=np.float64)
        self.predicted_traj: Optional[np.ndarray] = None
        self.phase: int = 1  # 1: High-level NMPC, 2: Terminal Flare Touchdown
        self.solve_time_history = []

        # Explicit solver statistics counters
        self.nmpc_calls: int = 0
        self.nmpc_success_count: int = 0
        self.nmpc_fallback_count: int = 0
        self.flare_steps: int = 0

    def reset(self) -> None:
        """Resets internal warm-start buffers, states, and counters."""
        self.w_opt_prev = None
        self.last_mpc_time = -1.0
        self.current_a_thrust_cmd = np.array([0.0, 0.0, self.gravity], dtype=np.float64)
        self.predicted_traj = None
        self.phase = 1
        self.solve_time_history.clear()
        self.nmpc_calls = 0
        self.nmpc_success_count = 0
        self.nmpc_fallback_count = 0
        self.flare_steps = 0

    def get_solver_statistics(self) -> Dict[str, Any]:
        """Returns comprehensive NMPC optimization and fallback reliability metrics."""
        success_rate = (self.nmpc_success_count / self.nmpc_calls * 100.0) if self.nmpc_calls > 0 else 0.0
        fallback_rate = (self.nmpc_fallback_count / self.nmpc_calls * 100.0) if self.nmpc_calls > 0 else 0.0
        mean_time = float(np.mean(self.solve_time_history)) if len(self.solve_time_history) > 0 else 0.0
        max_time = float(np.max(self.solve_time_history)) if len(self.solve_time_history) > 0 else 0.0

        return {
            'total_calls': self.nmpc_calls,
            'success_count': self.nmpc_success_count,
            'success_rate_pct': success_rate,
            'fallback_count': self.nmpc_fallback_count,
            'fallback_rate_pct': fallback_rate,
            'flare_steps': self.flare_steps,
            'mean_solve_time_ms': mean_time,
            'max_solve_time_ms': max_time
        }

    def compute_control(
        self,
        state: Dict[str, np.ndarray],
        t: float,
        dt: float
    ) -> Dict[str, Any]:
        """
        Computes control commands for the given state using Receding Horizon NMPC + SE(3).
        """
        pos = state['p']
        vel = state['v']
        R_curr = state['R']
        omega_curr = state['omega']

        # 1. Check Terminal Flare Handover Condition
        # Handover to terminal precision flare landing when centered over pad (radial <= 0.25m)
        # and at low altitude (z <= 0.80m).
        radial_dist = float(np.linalg.norm(pos[:2] - self.target_pos[:2]))
        if self.phase == 1 and (pos[2] <= 0.80 and radial_dist <= 0.25):
            self.phase = 2

        if self.phase == 2:
            self.flare_steps += 1
            # Terminal precision flare landing: smooth sink rate and pad centering
            e_xy = self.target_pos[:2] - pos[:2]

            ax = 3.5 * e_xy[0] - 3.0 * vel[0]
            ay = 3.5 * e_xy[1] - 3.0 * vel[1]

            vz_des = -float(np.clip(0.12 + 0.40 * pos[2], 0.12, 0.6))
            az = 4.5 * (vz_des - vel[2]) + self.gravity
            az = max(az, 1.0)

            self.current_a_thrust_cmd = np.array([ax, ay, az], dtype=np.float64)
            nmpc_info = {
                'status': 'FLARE',
                'solve_time_ms': 0.0,
                'cost': 0.0
            }
        else:
            # 2. High-level Receding Horizon NMPC Update (dt_mpc interval)
            if (t - self.last_mpc_time >= self.dt_mpc - 1e-6) or (self.last_mpc_time < 0.0):
                x_curr = np.concatenate([pos, vel])

                # Glideslope Reference Governor:
                # Maintains safe altitude buffer while transiting horizontally to prevent premature ground contact
                if radial_dist > 0.60:
                    target_z = float(np.clip(0.35 * radial_dist, 0.80, 10.0))
                    target_vz = -0.50 if pos[2] > target_z else 0.0
                    x_targ = np.array([self.target_pos[0], self.target_pos[1], target_z, 0.0, 0.0, target_vz], dtype=np.float64)
                else:
                    x_targ = np.concatenate([self.target_pos, self.target_vel])

                # Warm-start shift if previous solution exists
                w_guess = None
                if self.w_opt_prev is not None:
                    w_guess = self.nmpc_model.shift_solution(self.w_opt_prev)

                # Solve OCP NLP via CasADi + IPOPT
                self.nmpc_calls += 1
                success, u0, w_opt, nmpc_info = self.nmpc_model.solve(
                    x_init=x_curr,
                    x_target=x_targ,
                    w_guess=w_guess
                )

                if success:
                    self.nmpc_success_count += 1
                    self.current_a_thrust_cmd = u0
                    self.w_opt_prev = w_opt
                    self.predicted_traj = nmpc_info['X_opt'][:, :3]
                else:
                    # Emergency fallback: ZEM/ZEV guidance
                    self.nmpc_fallback_count += 1
                    t_go_est = max(1.0, float(np.linalg.norm(pos - self.target_pos) / (np.linalg.norm(vel) + 1.0)))
                    self.current_a_thrust_cmd = self.fallback_guidance.compute_thrust_acceleration(
                        p=pos,
                        v=vel,
                        pf=self.target_pos,
                        vf=self.target_vel,
                        t_go=t_go_est
                    )

                self.solve_time_history.append(nmpc_info['solve_time_ms'])
                self.last_mpc_time = t
            else:
                nmpc_info = {
                    'status': 'HOLD',
                    'solve_time_ms': 0.0,
                    'cost': 0.0
                }

        # 3. Inner Loop: SE(3) Geometric Attitude and Thrust Tracking (200 Hz)
        thrust, torque, debug_se3 = self.att_ctrl.compute_control(
            R_curr=R_curr,
            omega_curr=omega_curr,
            a_thrust_des=self.current_a_thrust_cmd,
            yaw_des=0.0
        )

        u_raw = np.array([thrust, torque[0], torque[1], torque[2]], dtype=np.float64)

        succ_rate = (self.nmpc_success_count / self.nmpc_calls * 100.0) if self.nmpc_calls > 0 else 0.0
        fall_rate = (self.nmpc_fallback_count / self.nmpc_calls * 100.0) if self.nmpc_calls > 0 else 0.0

        return {
            'thrust': thrust,
            'torque': torque,
            'u_raw': u_raw,
            'phase': self.phase,
            'target_pos': self.target_pos,
            'info': {
                'nmpc_a_cmd': self.current_a_thrust_cmd,
                'predicted_trajectory': self.predicted_traj,
                'solve_time_ms': nmpc_info.get('solve_time_ms', 0.0),
                'solver_status': nmpc_info.get('status', 'HOLD'),
                'tilt_deg': debug_se3['tilt_deg'],
                'e_R': debug_se3['e_R'],
                'e_Omega': debug_se3['e_Omega'],
                'total_nmpc_calls': self.nmpc_calls,
                'nmpc_success_count': self.nmpc_success_count,
                'nmpc_fallback_count': self.nmpc_fallback_count,
                'nmpc_success_rate': succ_rate,
                'nmpc_fallback_rate': fall_rate
            }
        }
