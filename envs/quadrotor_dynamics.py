"""
Quadrotor 6-DOF Nonlinear Rigid-Body Flight Dynamics Simulation Engine
Inertial Frame: ENU (East-North-Up)
Body Frame: Forward-Left-Up (+Z_B aligned with total thrust vector)

Author: Sungcheol Ha (KAIST Flight Dynamics & GNC Research Project)
"""

import numpy as np
import yaml
from typing import Dict, Tuple, Optional, Any


class Quadrotor6DOF:
    """
    Nonlinear 6-DOF Quadrotor Dynamics with 4th-Order Runge-Kutta (RK4) Integration.
    State Vector x in R^13:
        p: [x, y, z]            Inertial position [m]
        v: [vx, vy, vz]         Inertial velocity [m/s]
        q: [qw, qx, qy, qz]     Unit quaternion representing rotation from Body to Inertial (R_WB)
        omega: [p, q, r]        Body-fixed angular rates [rad/s]
    Control Vector u in R^4:
        T: Total collective thrust [N]
        tau: Body control torques [tau_x, tau_y, tau_z] [N*m]
    """

    def __init__(self, config_or_path: Any = None):
        if isinstance(config_or_path, str):
            with open(config_or_path, 'r') as f:
                self.cfg = yaml.safe_load(f)
        elif isinstance(config_or_path, dict):
            self.cfg = config_or_path
        else:
            # Default fallback parameters (AE300 Model)
            self.cfg = {
                'mass': 31.0,
                'gravity': 9.80665,
                'inertia': {'Ixx': 1.25, 'Iyy': 1.25, 'Izz': 2.50},
                'geometry': {
                    'arm_length': 0.4,
                    'thrust_max': 600.0,
                    'thrust_min': 0.0,
                    'torque_max': {'roll': 60.0, 'pitch': 60.0, 'yaw': 20.0},
                    'tilt_max_deg': 35.0
                },
                'drag': {'dx': 0.1, 'dy': 0.1, 'dz': 0.1}
            }

        # Physical constants
        self.mass: float = float(self.cfg['mass'])
        self.gravity: float = float(self.cfg['gravity'])
        self.g_vec: np.ndarray = np.array([0.0, 0.0, -self.gravity], dtype=np.float64)

        # Inertia tensor & inverse
        Ixx = float(self.cfg['inertia']['Ixx'])
        Iyy = float(self.cfg['inertia']['Iyy'])
        Izz = float(self.cfg['inertia']['Izz'])
        self.J: np.ndarray = np.diag([Ixx, Iyy, Izz]).astype(np.float64)
        self.J_inv: np.ndarray = np.diag([1.0 / Ixx, 1.0 / Iyy, 1.0 / Izz]).astype(np.float64)

        # Actuator limits
        geom = self.cfg.get('geometry', {})
        self.thrust_max: float = float(geom.get('thrust_max', 600.0))
        self.thrust_min: float = float(geom.get('thrust_min', 0.0))
        torq_cfg = geom.get('torque_max', {})
        self.torque_max: np.ndarray = np.array([
            float(torq_cfg.get('roll', 60.0)),
            float(torq_cfg.get('pitch', 60.0)),
            float(torq_cfg.get('yaw', 20.0))
        ], dtype=np.float64)

        # Drag damping matrix
        drag = self.cfg.get('drag', {})
        self.D_v: np.ndarray = np.diag([
            float(drag.get('dx', 0.1)),
            float(drag.get('dy', 0.1)),
            float(drag.get('dz', 0.1))
        ]).astype(np.float64)

        # Internal state
        self.state: np.ndarray = np.zeros(13, dtype=np.float64)
        self.time: float = 0.0
        self.contact_ground: bool = False

    @staticmethod
    def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
        """
        Converts unit quaternion [qw, qx, qy, qz] to 3x3 rotation matrix R_WB.
        v_W = R_WB * v_B
        """
        qw, qx, qy, qz = q
        # Ensure normalization protection
        norm_sq = qw * qw + qx * qx + qy * qy + qz * qz
        s = 2.0 / norm_sq if norm_sq > 1e-12 else 0.0

        R = np.empty((3, 3), dtype=np.float64)
        R[0, 0] = 1.0 - s * (qy * qy + qz * qz)
        R[0, 1] = s * (qx * qy - qz * qw)
        R[0, 2] = s * (qx * qz + qy * qw)

        R[1, 0] = s * (qx * qy + qz * qw)
        R[1, 1] = 1.0 - s * (qx * qx + qz * qz)
        R[1, 2] = s * (qy * qz - qx * qw)

        R[2, 0] = s * (qx * qz - qy * qw)
        R[2, 1] = s * (qy * qz + qx * qw)
        R[2, 2] = 1.0 - s * (qx * qx + qy * qy)
        return R

    @staticmethod
    def quat_derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """
        Calculates time derivative of unit quaternion:
        q_dot = 0.5 * q (x) [0, omega]^T
        """
        qw, qx, qy, qz = q
        p, q_rate, r = omega

        dq = np.empty(4, dtype=np.float64)
        dq[0] = 0.5 * (-qx * p - qy * q_rate - qz * r)
        dq[1] = 0.5 * (qw * p + qy * r - qz * q_rate)
        dq[2] = 0.5 * (qw * q_rate - qx * r + qz * p)
        dq[3] = 0.5 * (qw * r + qx * q_rate - qy * p)
        return dq

    def reset(
        self,
        init_pos: np.ndarray,
        init_vel: Optional[np.ndarray] = None,
        init_q: Optional[np.ndarray] = None,
        init_omega: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """Resets the simulation state."""
        self.state.fill(0.0)
        self.state[0:3] = np.asarray(init_pos, dtype=np.float64)

        if init_vel is not None:
            self.state[3:6] = np.asarray(init_vel, dtype=np.float64)

        if init_q is not None:
            q_arr = np.asarray(init_q, dtype=np.float64)
            self.state[6:10] = q_arr / np.linalg.norm(q_arr)
        else:
            self.state[6] = 1.0  # qw = 1, upright attitude

        if init_omega is not None:
            self.state[10:13] = np.asarray(init_omega, dtype=np.float64)

        self.time = 0.0
        self.contact_ground = False
        return self.get_state_dict()

    def dynamics(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Calculates state derivative dx/dt for given state x and control u.
        x: [p(3), v(3), q(4), omega(3)]
        u: [T, tau_x, tau_y, tau_z]
        """
        p = x[0:3]
        v = x[3:6]
        q = x[6:10]
        omega = x[10:13]

        # Normalize quaternion for rotation evaluation
        q_norm = q / (np.linalg.norm(q) + 1e-12)
        R_WB = self.quat_to_rot_matrix(q_norm)

        # Control inputs with actuator clipping
        T = np.clip(u[0], self.thrust_min, self.thrust_max)
        tau = np.clip(u[1:4], -self.torque_max, self.torque_max)

        # 1. Translational kinematics
        p_dot = v

        # 2. Translational dynamics
        # Body thrust vector along +Z_B: f_B = [0, 0, T]^T
        thrust_inertial = R_WB @ np.array([0.0, 0.0, T], dtype=np.float64)
        drag_force = self.D_v @ v
        v_dot = self.g_vec + (thrust_inertial - drag_force) / self.mass

        # 3. Rotational kinematics
        q_dot = self.quat_derivative(q_norm, omega)

        # 4. Rotational dynamics: J * omega_dot = -omega x (J * omega) + tau
        gyroscopic = np.cross(omega, self.J @ omega)
        omega_dot = self.J_inv @ (tau - gyroscopic)

        dx = np.empty(13, dtype=np.float64)
        dx[0:3] = p_dot
        dx[3:6] = v_dot
        dx[6:10] = q_dot
        dx[10:13] = omega_dot
        return dx

    def step(self, u: np.ndarray, dt: float) -> Tuple[Dict[str, np.ndarray], bool, dict]:
        """
        Executes a single numerical integration step using 4th-Order Runge-Kutta (RK4).
        """
        x = self.state
        t = self.time

        # RK4 Integration stages
        k1 = self.dynamics(t, x, u)
        k2 = self.dynamics(t + 0.5 * dt, x + 0.5 * dt * k1, u)
        k3 = self.dynamics(t + 0.5 * dt, x + 0.5 * dt * k2, u)
        k4 = self.dynamics(t + dt, x + dt * k3, u)

        x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Enforce quaternion unit norm constraint
        q_norm = np.linalg.norm(x_next[6:10])
        if q_norm > 1e-12:
            x_next[6:10] /= q_norm

        self.time += dt

        # Ground boundary handling (Inelastic ground contact condition at z <= 0)
        done = False
        info = {'contact_ground': False, 'time': self.time}

        if x_next[2] <= 0.0:
            x_next[2] = 0.0
            x_next[3:6] = 0.0          # Linear velocity stops
            x_next[10:13] = 0.0        # Angular velocity stops
            self.contact_ground = True
            done = True
            info['contact_ground'] = True

        self.state = x_next
        return self.get_state_dict(), done, info

    def get_state_dict(self) -> Dict[str, np.ndarray]:
        """Returns structured state dictionary for controller and metrics consumption."""
        p = self.state[0:3].copy()
        v = self.state[3:6].copy()
        q = self.state[6:10].copy()
        omega = self.state[10:13].copy()
        R = self.quat_to_rot_matrix(q)
        return {
            'p': p,
            'v': v,
            'q': q,
            'R': R,
            'omega': omega,
            'time': self.time
        }

    def get_euler_angles(self) -> np.ndarray:
        """
        Converts current quaternion to Euler angles [roll, pitch, yaw] in radians.
        Order: Z-Y-X (yaw, pitch, roll)
        """
        qw, qx, qy, qz = self.state[6:10]
        # Roll (phi)
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (theta)
        sinp = 2.0 * (qw * qy - qz * qx)
        sinp = np.clip(sinp, -1.0 + 1e-7, 1.0 - 1e-7)
        pitch = np.arcsin(sinp)

        # Yaw (psi)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw], dtype=np.float64)
