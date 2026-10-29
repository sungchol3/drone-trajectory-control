"""
Unit Test and Sanity Check for Visualization and Metrics Utilities
Author: Sungcheol Ha (KAIST Flight Dynamics & GNC Research Project)
"""

import sys
import os
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.metrics import calculate_flight_metrics
from utils.visualization import plot_3d_trajectory, plot_flight_telemetry


def test_visualization_pipeline():
    print("\n" + "=" * 60)
    print(">>> Running Unit Test: Visualization Pipeline Sanity Check")
    print("=" * 60)

    # 1. Generate Synthetic 3-Phase Flight History (Total: 8.6 s, dt = 0.02 s)
    dt = 0.02
    time = np.arange(0.0, 8.6 + dt, dt)
    n = len(time)

    pos = np.zeros((n, 3))
    vel = np.zeros((n, 3))
    quat = np.zeros((n, 4))
    euler = np.zeros((n, 3))
    omega = np.zeros((n, 3))
    thrust = np.full(n, 304.0)
    torque = np.zeros((n, 3))
    phase = np.zeros(n, dtype=int)
    target_pos = np.zeros((n, 3))

    # Phase 1: 0.0s ~ 2.5s (Descent from z=10 to z=5 at x=-10)
    # Phase 2: 2.5s ~ 6.5s (XY travel from x=-10 to x=0 at z=5)
    # Phase 3: 6.5s ~ 8.6s (Vertical soft landing from z=5 to z=0)
    for i, t in enumerate(time):
        if t <= 2.5:
            # Phase 1
            phase[i] = 1
            alpha = t / 2.5
            pos[i] = np.array([-10.0, 0.0, 10.0 - 5.0 * alpha])
            vel[i] = np.array([0.0, 0.0, -2.0])
            target_pos[i] = np.array([-10.0, 0.0, 5.0])
            thrust[i] = 280.0
        elif t <= 6.5:
            # Phase 2
            phase[i] = 2
            alpha = (t - 2.5) / 4.0
            pos[i] = np.array([-10.0 + 10.0 * alpha, 0.0, 5.0])
            vel[i] = np.array([2.5, 0.0, 0.0])
            target_pos[i] = np.array([0.0, 0.0, 5.0])
            # Pitch forward 8 deg for forward acceleration
            pitch = np.deg2rad(-8.0 * np.sin(np.pi * alpha))
            euler[i] = np.array([0.0, pitch, 0.0])
            thrust[i] = 310.0
        else:
            # Phase 3
            phase[i] = 3
            alpha = (t - 6.5) / 2.1
            pos[i] = np.array([0.0, 0.0, max(5.0 * (1.0 - alpha), 0.0)])
            vel[i] = np.array([0.0, 0.0, -1.0 * (1.0 - alpha)])
            target_pos[i] = np.array([0.0, 0.0, 0.0])
            thrust[i] = 300.0

        # Construct quaternion from euler
        p_pitch = euler[i, 1]
        quat[i] = np.array([np.cos(p_pitch / 2.0), 0.0, np.sin(p_pitch / 2.0), 0.0])

    history = {
        'time': time,
        'pos': pos,
        'vel': vel,
        'quat': quat,
        'euler': euler,
        'omega': omega,
        'thrust': thrust,
        'torque': torque,
        'phase': phase,
        'target_pos': target_pos
    }

    # 2. Test Metric Calculation
    mission_cfg = {
        'waypoints': {'target_pos': [0.0, 0.0, 0.0]},
        'landing_criteria': {'pos_tolerance': 0.05, 'vel_tolerance': 0.20}
    }
    metrics = calculate_flight_metrics(history, mission_cfg)
    print("Computed Test Metrics:")
    for k, v in metrics.items():
        print(f"  - {k}: {v}")

    assert metrics['flight_time'] == 8.6, "Flight time mismatch"
    assert metrics['min_altitude'] >= 0.0, "Negative altitude detected"

    # 3. Test Plot 3D Trajectory
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    plot3d_path = os.path.join(out_dir, 'test_3d_trajectory.png')
    plot_3d_trajectory(history, mission_cfg=mission_cfg, save_path=plot3d_path, show=False)
    assert os.path.exists(plot3d_path), "3D plot file was not generated"
    print(f"[PASS] 3D Trajectory plot created successfully ({os.path.getsize(plot3d_path)} bytes)")

    # 4. Test Plot Telemetry
    drone_cfg = {
        'mass': 31.0,
        'gravity': 9.80665,
        'geometry': {'thrust_max': 600.0, 'tilt_max_deg': 35.0}
    }
    telemetry_path = os.path.join(out_dir, 'test_telemetry.png')
    plot_flight_telemetry(history, drone_cfg=drone_cfg, mission_cfg=mission_cfg, 
                          metrics=metrics, save_path=telemetry_path, show=False)
    assert os.path.exists(telemetry_path), "Telemetry plot file was not generated"
    print(f"[PASS] Flight Telemetry plot created successfully ({os.path.getsize(telemetry_path)} bytes)")

    print("\n" + "=" * 60)
    print("Visualization & Metrics Pipeline Verification: ALL PASSED!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    test_visualization_pipeline()
