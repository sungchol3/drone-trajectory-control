"""
Flight Performance and Precision Landing Metrics Calculator
Author: Sungcheol Ha (KAIST Flight Dynamics & GNC Research Project)
"""

import numpy as np
from typing import Dict, Any, Optional


def calculate_flight_metrics(
    history: Dict[str, np.ndarray],
    mission_cfg: Optional[Dict[str, Any]] = None
) -> Dict[str, float]:
    """
    Computes rigorous quantitative metrics for 6-DOF drone trajectory and landing evaluation.
    
    Metrics:
        - flight_time: Total time taken to achieve touchdown or reach final step [s]
        - pos_error: Final Euclidean distance to target position [m]
        - final_x, final_y, final_z: Final position coordinates [m]
        - touchdown_vz: Vertical velocity at touchdown moment [m/s]
        - touchdown_v_mag: Total linear velocity magnitude at touchdown [m/s]
        - final_tilt_deg: Final pitch/roll tilt angle [deg]
        - min_altitude: Lowest altitude reached during flight [m]
        - control_effort: Integral of control effort: int (T^2 + ||tau||^2) dt [N^2 * s]
        - total_impulse: Sum of input changes: int (||dT|| + ||dtau||) dt [N * s]
        - soft_landing_success: Boolean indicating if touchdown criteria (pos and vel) were met
    """
    time = history['time']
    pos = history['pos']
    vel = history['vel']
    euler = history.get('euler', None)
    thrust = history.get('thrust', None)
    torque = history.get('torque', None)

    target_pos = np.array([0.0, 0.0, 0.0])
    pos_tol = 0.05
    vel_tol = 0.20

    if mission_cfg is not None:
        wp = mission_cfg.get('waypoints', {})
        target_pos = np.array(wp.get('target_pos', [0.0, 0.0, 0.0]))
        lc = mission_cfg.get('landing_criteria', {})
        pos_tol = float(lc.get('pos_tolerance', 0.05))
        vel_tol = float(lc.get('vel_tolerance', 0.20))

    dt = np.mean(np.diff(time)) if len(time) > 1 else 0.005
    n_steps = len(time)

    # 1. Terminal State & Time
    final_idx = n_steps - 1
    final_time = float(time[final_idx])
    final_p = pos[final_idx]
    final_v = vel[final_idx]

    pos_error = float(np.linalg.norm(final_p - target_pos))
    touchdown_vz = float(final_v[2])
    touchdown_v_mag = float(np.linalg.norm(final_v))

    # 2. Final Attitude Tilt
    final_tilt_deg = 0.0
    if euler is not None:
        roll = euler[final_idx, 0]
        pitch = euler[final_idx, 1]
        final_tilt_deg = float(np.rad2deg(np.arccos(np.clip(np.cos(roll) * np.cos(pitch), -1.0, 1.0))))

    # 3. Minimum Altitude (Check for underground penetration)
    min_altitude = float(np.min(pos[:, 2]))

    # 4. Energy and Control Effort
    control_effort = 0.0
    if thrust is not None and torque is not None:
        effort_arr = (thrust ** 2) + np.sum(torque ** 2, axis=1)
        control_effort = float(np.trapz(effort_arr, time))

    # 5. Total Impulse (Smoothness metric)
    total_impulse = 0.0
    if thrust is not None:
        delta_T = np.abs(np.diff(thrust))
        delta_tau = np.sum(np.abs(np.diff(torque, axis=0)), axis=1) if torque is not None else 0.0
        total_impulse = float(np.sum(delta_T + delta_tau) * dt)

    # 6. Landing Criterion Check
    soft_landing_success = (pos_error <= pos_tol) and (touchdown_v_mag <= vel_tol) and (min_altitude >= -1e-4)

    return {
        'flight_time': round(final_time, 3),
        'pos_error': round(pos_error, 4),
        'final_x': round(float(final_p[0]), 4),
        'final_y': round(float(final_p[1]), 4),
        'final_z': round(float(final_p[2]), 4),
        'touchdown_vz': round(touchdown_vz, 4),
        'touchdown_v_mag': round(touchdown_v_mag, 4),
        'final_tilt_deg': round(final_tilt_deg, 2),
        'min_altitude': round(min_altitude, 4),
        'control_effort': round(control_effort, 2),
        'total_impulse': round(total_impulse, 4),
        'soft_landing_success': bool(soft_landing_success)
    }
