"""
Abstract Base Controller Interface
Author: Sungcheol Ha (KAIST Flight Dynamics & GNC Research Project)
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np


class BaseController(ABC):
    """
    Abstract Base Class for all multirotor guidance and tracking controllers
    (Cascaded PID, ZEM/ZEV, NMPC, RL).
    """

    @abstractmethod
    def reset(self) -> None:
        """Resets controller internal states (integrators, filters, FSM phases)."""
        pass

    @abstractmethod
    def compute_control(
        self,
        state: Dict[str, np.ndarray],
        t: float,
        dt: float
    ) -> Dict[str, Any]:
        """
        Computes control commands for the given current state.

        Args:
            state: Dictionary containing:
                - 'p': Position [x, y, z] in ENU frame [m] (3,)
                - 'v': Linear velocity [vx, vy, vz] [m/s] (3,)
                - 'q': Unit quaternion [qw, qx, qy, qz] (4,)
                - 'R': Rotation matrix R_WB (3, 3)
                - 'omega': Body angular velocity [p, q, r] [rad/s] (3,)
                - 'time': Current simulation time [s]
            t: Current time [s]
            dt: Control timestep [s]

        Returns:
            Dictionary containing:
                - 'thrust': Total collective thrust T [N] (float)
                - 'torque': Body control torque [tau_x, tau_y, tau_z] [N*m] (3,)
                - 'u_raw': Vector [T, tau_x, tau_y, tau_z] (4,)
                - 'phase': FSM phase integer (optional)
                - 'target_pos': Target position vector [x_d, y_d, z_d] (3,)
                - 'info': Arbitrary debugging dictionary
        """
        pass
