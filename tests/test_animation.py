"""
Unit Test for 3D Airframe Geometry & Flight Animation Module
Author: Sungcheol Ha (KAIST Flight Dynamics & GNC Research Project)
"""

import os
import sys
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.animation import DroneAirframeVisual, InteractiveFlightAnimator


def test_drone_airframe_visual_geometry():
    """Verifies that 6-DOF airframe geometry is correctly transformed by quaternion."""
    visual = DroneAirframeVisual(arm_length=0.4, rotor_radius=0.12)

    # 1. Identity quaternion (Upright attitude)
    pos = np.array([0.0, 0.0, 5.0])
    q_identity = np.array([1.0, 0.0, 0.0, 0.0])
    thrust = 304.0

    geo = visual.compute_world_geometry(pos, q_identity, thrust)

    # Check rotor 1 (+X arm front)
    assert np.allclose(geo['arm_x'][1], [0.4, 0.0, 5.0]), "Rotor 1 position mismatch at identity quaternion"
    # Check thrust vector points up (+Z)
    assert np.allclose(geo['b3'], [0.0, 0.0, 1.0]), "Thrust direction mismatch at identity quaternion"
    assert np.allclose(geo['thrust_end'], [0.0, 0.0, 5.8]), "Thrust end position mismatch"

    # 2. Pitch 90 deg about Y axis: q = [cos(pi/4), 0, sin(pi/4), 0]
    q_pitch90 = np.array([np.cos(np.pi/4), 0.0, np.sin(np.pi/4), 0.0])
    geo_pitch = visual.compute_world_geometry(pos, q_pitch90, thrust)
    # Rotor 1 (+X_B) should now point straight down (-Z_W)
    assert np.allclose(geo_pitch['arm_x'][1], [0.0, 0.0, 5.0 - 0.4], atol=1e-5), "Pitched arm transformation failed"
    # Thrust (+Z_B) should now point along +X_W
    assert np.allclose(geo_pitch['b3'], [1.0, 0.0, 0.0], atol=1e-5), "Pitched thrust vector failed"

    print("[PASS] test_drone_airframe_visual_geometry: 3D Airframe & Thrust transformations verified.")


def test_animation_headless_export():
    """Verifies headless video encoding pipeline on mock trajectory."""
    # Create short 1.0s mock history (50 timesteps)
    t = np.linspace(0, 1.0, 50)
    pos = np.zeros((50, 3))
    pos[:, 0] = -10.0 + 10.0 * (t / 1.0)
    pos[:, 2] = 10.0 - 5.0 * (t / 1.0)
    vel = np.zeros((50, 3))
    vel[:, 0] = 10.0
    vel[:, 2] = -5.0
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (50, 1))
    thrust = np.full(50, 304.0)
    phase = np.full(50, 2)
    euler = np.zeros((50, 3))

    history = {
        'time': t,
        'pos': pos,
        'vel': vel,
        'quat': quat,
        'thrust': thrust,
        'phase': phase,
        'euler': euler
    }

    test_out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'test_animation.mp4'))

    animator = InteractiveFlightAnimator(history=history, fps=20, speed_multiplier=2.0)
    animator.run(save_path=test_out_path, show=False)

    assert os.path.exists(test_out_path), "Encoded animation file does not exist"
    file_size = os.path.getsize(test_out_path)
    assert file_size > 1000, f"Encoded animation file too small: {file_size} bytes"

    # Clean up test artifact
    if os.path.exists(test_out_path):
        os.remove(test_out_path)

    print(f"[PASS] test_animation_headless_export: Video encoding verified successfully (size: {file_size} bytes).")


if __name__ == "__main__":
    test_drone_airframe_visual_geometry()
    test_animation_headless_export()
