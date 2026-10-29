"""
Unit Tests for Quadrotor 6-DOF Dynamics Engine
Scenario 1: Free Fall Dynamics and Mechanical Energy Conservation Test
Scenario 2: Static Hovering Equilibrium State Test

Author: Sungcheol Ha (KAIST Flight Dynamics & GNC Research Project)
"""

import sys
import os
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.quadrotor_dynamics import Quadrotor6DOF


def test_scenario_1_free_fall_energy_conservation():
    """
    Scenario 1: Free Fall and Energy Conservation
    - Condition: Zero thrust, zero torque, zero aerodynamic drag
    - Initial: Height z0 = 10.0 m, v0 = 0.0 m/s
    - Duration: 1.0 s
    - Verification:
        1. Analytic position z(t) = z0 - 0.5 * g * t^2
        2. Analytic velocity vz(t) = -g * t
        3. Mechanical energy E(t) = m*g*z + 0.5*m*v^2 is conserved within 1e-5 relative error.
    """
    print("\n" + "=" * 60)
    print(">>> Running Unit Test: Scenario 1 - Free Fall & Energy Conservation")
    print("=" * 60)

    # Initialize model with zero aerodynamic drag for pure conservative physics test
    config = {
        'mass': 31.0,
        'gravity': 9.80665,
        'inertia': {'Ixx': 1.25, 'Iyy': 1.25, 'Izz': 2.50},
        'geometry': {
            'arm_length': 0.4,
            'thrust_max': 600.0,
            'thrust_min': 0.0,
            'torque_max': {'roll': 60.0, 'pitch': 60.0, 'yaw': 20.0}
        },
        'drag': {'dx': 0.0, 'dy': 0.0, 'dz': 0.0}  # Pure conservative field
    }
    env = Quadrotor6DOF(config)

    z0 = 10.0
    env.reset(init_pos=np.array([0.0, 0.0, z0]))

    m = env.mass
    g = env.gravity
    E0 = m * g * z0  # Initial potential energy

    dt = 0.005
    total_time = 1.0
    steps = int(total_time / dt)

    u_zero = np.array([0.0, 0.0, 0.0, 0.0])

    for step in range(steps):
        state, done, info = env.step(u=u_zero, dt=dt)
        if done:
            break

    final_z = state['p'][2]
    final_vz = state['v'][2]
    sim_time = state['time']

    # Analytic solutions
    z_theory = z0 - 0.5 * g * (sim_time ** 2)
    vz_theory = -g * sim_time
    E_final = m * g * final_z + 0.5 * m * (final_vz ** 2)

    pos_error = abs(final_z - z_theory)
    vel_error = abs(final_vz - vz_theory)
    energy_rel_error = abs(E_final - E0) / E0

    print(f"Simulation Time:    {sim_time:.3f} s")
    print(f"Simulated Position: z = {final_z:.6f} m  (Theory: {z_theory:.6f} m, Error: {pos_error:.3e} m)")
    print(f"Simulated Velocity: vz = {final_vz:.6f} m/s (Theory: {vz_theory:.6f} m/s, Error: {vel_error:.3e} m/s)")
    print(f"Initial Energy:     {E0:.6f} J")
    print(f"Final Energy:       {E_final:.6f} J (Relative Error: {energy_rel_error:.3e})")

    assert pos_error < 1e-4, f"Position error too large: {pos_error:.3e} m"
    assert vel_error < 1e-4, f"Velocity error too large: {vel_error:.3e} m/s"
    assert energy_rel_error < 1e-5, f"Energy conservation violated: {energy_rel_error:.3e}"

    print("[PASS] Scenario 1: Free Fall and Energy Conservation Verified Successfully.")
    return True


def test_scenario_2_static_hover_equilibrium():
    """
    Scenario 2: Static Hovering Equilibrium State
    - Condition: Initial position [-10, 0, 10], zero velocity, upright attitude
    - Control: Exact gravity compensation thrust T_hover = m * g, zero torques
    - Duration: 10.0 s (2,000 steps at dt = 0.005 s)
    - Verification:
        1. Position drift ||Delta p|| < 1e-4 m
        2. Velocity drift ||v|| < 1e-4 m/s
        3. Attitude deviation ||q - [1, 0, 0, 0]|| < 1e-6
    """
    print("\n" + "=" * 60)
    print(">>> Running Unit Test: Scenario 2 - Static Hovering Equilibrium")
    print("=" * 60)

    # Standard configuration with drag
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'configs', 'drone_params.yaml'))
    env = Quadrotor6DOF(config_path)

    init_pos = np.array([-10.0, 0.0, 10.0])
    env.reset(init_pos=init_pos)

    # Hover equilibrium thrust
    T_hover = env.mass * env.gravity
    u_hover = np.array([T_hover, 0.0, 0.0, 0.0])

    dt = 0.005
    total_time = 10.0
    steps = int(total_time / dt)

    print(f"Mass: {env.mass} kg, Gravity: {env.gravity} m/s^2")
    print(f"Hover Equilibrium Thrust Applied: T = {T_hover:.4f} N")

    for _ in range(steps):
        state, done, info = env.step(u=u_hover, dt=dt)
        if done:
            break

    final_pos = state['p']
    final_vel = state['v']
    final_q = state['q']
    pos_drift = np.linalg.norm(final_pos - init_pos)
    vel_drift = np.linalg.norm(final_vel)
    quat_err = np.linalg.norm(final_q - np.array([1.0, 0.0, 0.0, 0.0]))

    print(f"Simulation Duration: {state['time']:.2f} s ({steps} RK4 steps)")
    print(f"Initial Position:    {init_pos}")
    print(f"Final Position:      {final_pos} (Drift: {pos_drift:.3e} m)")
    print(f"Final Velocity:      {final_vel} (Magnitude: {vel_drift:.3e} m/s)")
    print(f"Final Quaternion:    {final_q} (Deviation: {quat_err:.3e})")

    assert pos_drift < 1e-4, f"Position drift exceeded threshold: {pos_drift:.3e} m"
    assert vel_drift < 1e-4, f"Velocity drift exceeded threshold: {vel_drift:.3e} m/s"
    assert quat_err < 1e-6, f"Attitude deviation exceeded threshold: {quat_err:.3e}"

    print("[PASS] Scenario 2: Static Hovering Equilibrium Verified Successfully.")
    return True


if __name__ == "__main__":
    t1_pass = test_scenario_1_free_fall_energy_conservation()
    t2_pass = test_scenario_2_static_hover_equilibrium()

    print("\n" + "=" * 60)
    print("All Target Unit Tests (Scenario 1 & 2) Passed Successfully!")
    print("=" * 60)
