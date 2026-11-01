"""
CasADi Symbolic NMPC Model and Optimal Control Problem (OCP) Solver
High-Fidelity 6-DOF Multirotor Precision Landing with Hard State & Actuator Constraints

Features:
  - Symbolic 6-DOF translational dynamics with aerodynamic drag
  - 4th-order Runge-Kutta (RK4) multiple shooting discretization
  - Hard inequality constraints:
      * Ground collision avoidance: z_k >= z_min (buffer 0.04m)
      * Geometric tilt cone: ||u_xy|| <= u_z * tan(theta_max)
      * Maximum collective thrust saturation: ||u|| <= T_max / m
      * Terminal soft touchdown velocity constraint: vz_N >= -v_touchdown
  - Fast Multiple Shooting NLP solved via CasADi + IPOPT
  - Efficient warm-starting support for sub-15ms real-time execution

Author: Sungcheol Ha (KAIST Flight Dynamics & GNC Research Project)
"""

import casadi as ca
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, Optional


@dataclass
class NMPCConfig:
    """Configuration parameters for CasADi NMPC OCP solver."""
    # Horizon and Discretization
    N: int = 25                             # Prediction horizon steps
    dt_mpc: float = 0.04                    # Timestep per step [s] (T_horizon = 1.0s)

    # Physical vehicle parameters
    mass: float = 31.0                      # Total mass [kg]
    gravity: float = 9.80665                # Gravitational acceleration [m/s^2]
    thrust_min: float = 0.0                 # Minimum collective thrust [N]
    thrust_max: float = 600.0               # Maximum collective thrust [N]
    tilt_max_deg: float = 35.0              # Maximum tilt angle constraint [deg]
    drag_coeffs: np.ndarray = field(        # Linear drag coefficients [N*s/m]
        default_factory=lambda: np.array([0.1, 0.1, 0.1], dtype=np.float64)
    )

    # Hard constraints
    z_min: float = 0.04                     # Ground clearance buffer [m]
    v_max: float = 12.0                     # Maximum velocity magnitude [m/s]
    v_touchdown: float = 0.20               # Maximum terminal touchdown sink rate [m/s]
    a_z_min: float = 1.0                    # Minimum positive vertical thrust accel [m/s^2]

    # Objective weights (diagonal matrices)
    Q_pos: np.ndarray = field(              # Stage position error weight
        default_factory=lambda: np.array([30.0, 30.0, 30.0], dtype=np.float64)
    )
    Q_vel: np.ndarray = field(              # Stage velocity error weight
        default_factory=lambda: np.array([5.0, 5.0, 5.0], dtype=np.float64)
    )
    R_thrust: np.ndarray = field(           # Control acceleration effort weight
        default_factory=lambda: np.array([0.1, 0.1, 0.1], dtype=np.float64)
    )
    R_dthrust: np.ndarray = field(          # Control jerk / rate change weight
        default_factory=lambda: np.array([0.5, 0.5, 0.5], dtype=np.float64)
    )
    P_pos: np.ndarray = field(              # Terminal position error weight
        default_factory=lambda: np.array([100.0, 100.0, 100.0], dtype=np.float64)
    )
    P_vel: np.ndarray = field(              # Terminal velocity error weight
        default_factory=lambda: np.array([20.0, 20.0, 20.0], dtype=np.float64)
    )

    # IPOPT solver options
    max_iter: int = 150
    tol: float = 1e-4
    print_level: int = 0                    # 0 to suppress IPOPT verbosity


class CasADiNMPCModel:
    """
    Symbolic Multiple Shooting OCP Builder and IPOPT Solver Wrapper.
    """

    def __init__(self, cfg: Optional[NMPCConfig] = None):
        self.cfg = cfg if cfg is not None else NMPCConfig()

        self.nx = 6   # States: [p_x, p_y, p_z, v_x, v_y, v_z]
        self.nu = 3   # Controls: [a_x, a_y, a_z] (thrust acceleration in ENU)
        self.N = self.cfg.N
        self.dt = self.cfg.dt_mpc

        self.max_accel = self.cfg.thrust_max / self.cfg.mass  # ~19.35 m/s^2
        self.tan_tilt_max = np.tan(np.deg2rad(self.cfg.tilt_max_deg))

        # Build CasADi symbolic NLP graph and solver
        self._build_nlp()

    def _build_nlp(self) -> None:
        """Constructs multiple shooting NLP with RK4 integration and hard constraints."""
        # 1. Symbolic Continuous Dynamics Function: \dot{x} = f(x, u)
        x_sym = ca.SX.sym('x', self.nx)
        u_sym = ca.SX.sym('u', self.nu)

        p = x_sym[:3]
        v = x_sym[3:]
        a_thrust = u_sym

        # Gravity and aerodynamic drag
        g_vec = ca.vertcat(0.0, 0.0, -self.cfg.gravity)
        d_vec = ca.vertcat(
            float(self.cfg.drag_coeffs[0]),
            float(self.cfg.drag_coeffs[1]),
            float(self.cfg.drag_coeffs[2])
        )
        drag_force = (d_vec / self.cfg.mass) * v

        p_dot = v
        v_dot = a_thrust + g_vec - drag_force
        x_dot = ca.vertcat(p_dot, v_dot)

        f_dyn = ca.Function('f_dyn', [x_sym, u_sym], [x_dot])

        # 2. Symbolic RK4 Integrator: x_{k+1} = F_RK4(x_k, u_k, dt)
        k1 = f_dyn(x_sym, u_sym)
        k2 = f_dyn(x_sym + 0.5 * self.dt * k1, u_sym)
        k3 = f_dyn(x_sym + 0.5 * self.dt * k2, u_sym)
        k4 = f_dyn(x_sym + self.dt * k3, u_sym)
        x_next_rk4 = x_sym + (self.dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        self.f_rk4 = ca.Function('f_rk4', [x_sym, u_sym], [x_next_rk4])

        # 3. Decision Variables for Multiple Shooting
        # w = [x_0, u_0, x_1, u_1, ..., x_{N-1}, u_{N-1}, x_N]
        X = ca.SX.sym('X', self.nx, self.N + 1)
        U = ca.SX.sym('U', self.nu, self.N)

        # Parameter Vector: P = [x_current (6), x_target (6)]
        P = ca.SX.sym('P', 12)
        x_curr = P[:6]
        x_targ = P[6:]

        cost = 0.0
        g = []
        lbg = []
        ubg = []

        # Reference hover thrust acceleration (balances gravity)
        u_hover = ca.vertcat(0.0, 0.0, self.cfg.gravity)

        # 4. Initial Condition Constraint: x_0 == x_current
        g.append(X[:, 0] - x_curr)
        lbg.extend([0.0] * self.nx)
        ubg.extend([0.0] * self.nx)

        # 5. Multiple Shooting Horizon Loops
        for k in range(self.N):
            x_k = X[:, k]
            u_k = U[:, k]
            p_k = x_k[:3]
            v_k = x_k[3:]

            # 5-1. Stage Cost
            # Position tracking error
            e_p = p_k - x_targ[:3]
            cost += ca.mtimes([e_p.T, ca.diag(self.cfg.Q_pos), e_p])

            # Velocity tracking error
            e_v = v_k - x_targ[3:]
            cost += ca.mtimes([e_v.T, ca.diag(self.cfg.Q_vel), e_v])

            # Control effort (deviation from hover)
            e_u = u_k - u_hover
            cost += ca.mtimes([e_u.T, ca.diag(self.cfg.R_thrust), e_u])

            # Control rate of change (jerk smoothing)
            if k > 0:
                delta_u = u_k - U[:, k - 1]
                cost += ca.mtimes([delta_u.T, ca.diag(self.cfg.R_dthrust), delta_u])

            # 5-2. Shooting Defect Constraint: x_{k+1} == F_RK4(x_k, u_k)
            x_pred = self.f_rk4(x_k, u_k)
            g.append(X[:, k + 1] - x_pred)
            lbg.extend([0.0] * self.nx)
            ubg.extend([0.0] * self.nx)

            # 5-3. Hard Inequality: Ground Collision Avoidance (z_k >= z_min)
            g.append(x_k[2])
            lbg.append(self.cfg.z_min)
            ubg.append(ca.inf)

            # 5-4. Hard Inequality: Tilt Cone Constraint
            # ||u_xy|| <= u_z * tan(theta_max)  ==>  u_x^2 + u_y^2 - u_z^2 * tan^2(theta_max) <= 0
            tilt_slack = u_k[0]**2 + u_k[1]**2 - (u_k[2] * self.tan_tilt_max)**2
            g.append(tilt_slack)
            lbg.append(-ca.inf)
            ubg.append(0.0)

            # 5-5. Hard Inequality: Maximum Thrust Bound
            # ||u||^2 <= (T_max / m)^2
            thrust_sq = u_k[0]**2 + u_k[1]**2 + u_k[2]**2
            g.append(thrust_sq)
            lbg.append(0.0)
            ubg.append(float(self.max_accel**2))

            # 5-6. Hard Inequality: Positive Vertical Lift
            g.append(u_k[2])
            lbg.append(self.cfg.a_z_min)
            ubg.append(float(self.max_accel))

            # 5-7. Velocity Magnitude Bound
            v_sq = v_k[0]**2 + v_k[1]**2 + v_k[2]**2
            g.append(v_sq)
            lbg.append(0.0)
            ubg.append(float(self.cfg.v_max**2))

        # 6. Terminal Cost & Constraints at k = N
        x_N = X[:, self.N]
        p_N = x_N[:3]
        v_N = x_N[3:]

        # Terminal position and velocity cost
        e_p_N = p_N - x_targ[:3]
        e_v_N = v_N - x_targ[3:]
        cost += ca.mtimes([e_p_N.T, ca.diag(self.cfg.P_pos), e_p_N])
        cost += ca.mtimes([e_v_N.T, ca.diag(self.cfg.P_vel), e_v_N])

        # Terminal Ground Clearance
        g.append(x_N[2])
        lbg.append(self.cfg.z_min)
        ubg.append(ca.inf)

        # Terminal Touchdown Soft Landing Sink Rate Constraint:
        # v_z,N >= -v_touchdown (-0.20 m/s)
        g.append(v_N[2])
        lbg.append(-float(self.cfg.v_touchdown))
        ubg.append(0.05)  # Slight buffer for hovering touchdown

        # 7. Flatten Decision Variables Vector: w
        # Ordering: [x_0, u_0, x_1, u_1, ..., x_{N-1}, u_{N-1}, x_N]
        w = []
        for k in range(self.N):
            w.append(X[:, k])
            w.append(U[:, k])
        w.append(X[:, self.N])
        w_vec = ca.vertcat(*w)

        # 8. Variable Upper and Lower Bounds (Box Bounds on w)
        lbw = []
        ubw = []
        for k in range(self.N):
            # State bounds [px, py, pz, vx, vy, vz]
            lbw.extend([-50.0, -50.0, 0.0, -self.cfg.v_max, -self.cfg.v_max, -self.cfg.v_max])
            ubw.extend([50.0, 50.0, 50.0, self.cfg.v_max, self.cfg.v_max, self.cfg.v_max])
            # Input bounds [ax, ay, az]
            lbw.extend([-self.max_accel, -self.max_accel, self.cfg.a_z_min])
            ubw.extend([self.max_accel, self.max_accel, self.max_accel])
        # Final state bounds
        lbw.extend([-50.0, -50.0, 0.0, -self.cfg.v_max, -self.cfg.v_max, -self.cfg.v_max])
        ubw.extend([50.0, 50.0, 50.0, self.cfg.v_max, self.cfg.v_max, self.cfg.v_max])

        self.lbw = np.array(lbw, dtype=np.float64)
        self.ubw = np.array(ubw, dtype=np.float64)
        self.lbg = np.array(lbg, dtype=np.float64)
        self.ubg = np.array(ubg, dtype=np.float64)
        self.w_dim = w_vec.shape[0]

        # 9. Instantiate IPOPT Solver
        nlp = {
            'x': w_vec,
            'f': cost,
            'g': ca.vertcat(*g),
            'p': P
        }

        ipopt_opts = {
            'print_time': False,
            'ipopt.print_level': self.cfg.print_level,
            'ipopt.sb': 'yes',                    # Suppress IPOPT startup banner
            'ipopt.max_iter': self.cfg.max_iter,
            'ipopt.tol': self.cfg.tol,
            'ipopt.acceptable_tol': self.cfg.tol * 10,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.hessian_approximation': 'limited-memory',  # L-BFGS for fast QP iterations
            'ipopt.linear_solver': 'mumps'
        }

        self.solver = ca.nlpsol('nmpc_solver', 'ipopt', nlp, ipopt_opts)

    def unpack_solution(self, w_sol: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Unpacks flat decision vector into state trajectory X (N+1, 6) and input trajectory U (N, 3).
        """
        w = np.asarray(w_sol, dtype=np.float64).flatten()
        X_sol = np.zeros((self.N + 1, self.nx))
        U_sol = np.zeros((self.N, self.nu))

        stride = self.nx + self.nu
        for k in range(self.N):
            idx = k * stride
            X_sol[k] = w[idx:idx + self.nx]
            U_sol[k] = w[idx + self.nx:idx + stride]
        X_sol[self.N] = w[self.N * stride:self.N * stride + self.nx]

        return X_sol, U_sol

    def pack_solution(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        """Packs state trajectory X (N+1, 6) and input trajectory U (N, 3) into flat vector."""
        w = []
        for k in range(self.N):
            w.extend(X[k])
            w.extend(U[k])
        w.extend(X[self.N])
        return np.array(w, dtype=np.float64)

    def shift_solution(self, w_sol: np.ndarray) -> np.ndarray:
        """
        Shifts previous optimal solution forward by 1 step for Warm-Starting:
        X_init = [x_1, x_2, ..., x_N, x_N]
        U_init = [u_1, u_2, ..., u_{N-1}, u_{N-1}]
        """
        X_sol, U_sol = self.unpack_solution(w_sol)

        # Shift forward
        X_shift = np.zeros_like(X_sol)
        U_shift = np.zeros_like(U_sol)

        X_shift[:-1] = X_sol[1:]
        X_shift[-1] = X_sol[-1]

        U_shift[:-1] = U_sol[1:]
        U_shift[-1] = U_sol[-1]

        return self.pack_solution(X_shift, U_shift)

    def get_initial_guess(self, x_curr: np.ndarray, x_targ: np.ndarray) -> np.ndarray:
        """Generates linear interpolation initial guess for cold start."""
        X_guess = np.zeros((self.N + 1, self.nx))
        U_guess = np.zeros((self.N, self.nu))

        for k in range(self.N + 1):
            alpha = k / self.N
            X_guess[k] = (1.0 - alpha) * x_curr + alpha * x_targ

        # Hover thrust acceleration guess [0, 0, g]
        U_guess[:, 2] = self.cfg.gravity

        return self.pack_solution(X_guess, U_guess)

    def solve(
        self,
        x_init: np.ndarray,
        x_target: np.ndarray,
        w_guess: Optional[np.ndarray] = None
    ) -> Tuple[bool, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Solves the OCP NLP.

        Args:
            x_init: Current vehicle state [x, y, z, vx, vy, vz] (6,)
            x_target: Desired target state [xf, yf, zf, vxf, vyf, vzf] (6,)
            w_guess: Optional warm-start initial guess vector

        Returns:
            success: Whether IPOPT converged successfully
            u_opt_0: Immediate optimal acceleration command a*_thrust,0 [m/s^2] (3,)
            w_opt: Optimal solution vector of all decision variables
            info: Diagnostics dictionary (X_opt, U_opt, cost, status)
        """
        import time

        x_init = np.asarray(x_init, dtype=np.float64).flatten()
        x_target = np.asarray(x_target, dtype=np.float64).flatten()

        P_val = np.concatenate([x_init, x_target])

        if w_guess is None:
            w0 = self.get_initial_guess(x_init, x_target)
        else:
            w0 = w_guess

        t_start = time.perf_counter()
        res = self.solver(
            x0=w0,
            lbx=self.lbw,
            ubx=self.ubw,
            lbg=self.lbg,
            ubg=self.ubg,
            p=P_val
        )
        t_solve = (time.perf_counter() - t_start) * 1000.0  # ms

        stats = self.solver.stats()
        status = stats['return_status']
        success = (status in ['Solve_Succeeded', 'Solved_To_Acceptable_Level'])

        w_opt = np.array(res['x']).flatten()
        cost_val = float(res['f'])

        X_opt, U_opt = self.unpack_solution(w_opt)
        u_opt_0 = U_opt[0].copy()

        info = {
            'success': success,
            'status': status,
            'cost': cost_val,
            'solve_time_ms': t_solve,
            'X_opt': X_opt,
            'U_opt': U_opt
        }

        return success, u_opt_0, w_opt, info
