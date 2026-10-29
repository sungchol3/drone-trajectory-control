"""
Scenario 3: 3-Phase Segmented PID Baseline Flight Simulation & Verification
Robust Point-to-Point Navigation and Precision Soft Landing

Features:
  - Multi-rate simulation (200 Hz RK4 physics + 50 Hz control loop)
  - 3-Phase sequential waypoint FSM with velocity profiling
  - Ground flare sink-rate regulation for soft touchdown
  - Automatic 3D & Telemetry static plot export
  - Interactive real-time 3D flight animation & telemetry GUI (CLI: --animate)
  - Optional headless video export (CLI: --save-anim)

Author: Sungcheol Ha (KAIST Flight Dynamics & GNC Research Project)
"""

import sys
import os
import argparse
import yaml
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from envs.quadrotor_dynamics import Quadrotor6DOF
from controllers.pid_controller import SegmentedPIDController
from utils.metrics import calculate_flight_metrics
from utils.visualization import plot_3d_trajectory, plot_flight_telemetry
from utils.animation import create_flight_animation


def run_baseline_pid_simulation(
    animate: bool = False,
    save_anim: bool = False,
    speed: float = 1.0,
    fps: int = 30
):
    print("\n" + "=" * 70)
    print(">>> Running Scenario 3: 3-Phase Segmented PID Baseline Simulation")
    print("=" * 70)

    # 1. Load Configurations
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    drone_cfg_path = os.path.join(root_dir, 'configs', 'drone_params.yaml')
    mission_cfg_path = os.path.join(root_dir, 'configs', 'mission_params.yaml')

    with open(drone_cfg_path, 'r') as f:
        drone_cfg = yaml.safe_load(f)
    with open(mission_cfg_path, 'r') as f:
        mission_cfg = yaml.safe_load(f)

    # 2. Instantiate Environment & Controller
    env = Quadrotor6DOF(drone_cfg)
    controller = SegmentedPIDController(drone_cfg, mission_cfg)

    # Mission Parameters
    start_pos = np.array(mission_cfg['waypoints']['start_pos'], dtype=np.float64)
    start_vel = np.array(mission_cfg['waypoints']['start_vel'], dtype=np.float64)
    target_pos = np.array(mission_cfg['waypoints']['target_pos'], dtype=np.float64)
    max_time = float(mission_cfg['timing']['max_time'])
    dt_sim = float(mission_cfg['timing']['dt_sim'])     # 0.005s (200 Hz)
    dt_ctrl = float(mission_cfg['timing']['dt_ctrl'])   # 0.020s (50 Hz)
    ctrl_substeps = max(int(dt_ctrl / dt_sim), 1)

    print(f"Vehicle Mass:          {env.mass:.1f} kg (30kg Drone + 1kg Payload)")
    print(f"Initial Position:      {start_pos} m")
    print(f"Target Landing Point:  {target_pos} m")
    print(f"Simulation Timestep:   dt_sim = {dt_sim*1000:.1f} ms, dt_ctrl = {dt_ctrl*1000:.1f} ms")

    # Reset
    state = env.reset(init_pos=start_pos, init_vel=start_vel)
    controller.reset()

    # 3. Data Collection Buffers
    history = {
        'time': [],
        'pos': [],
        'vel': [],
        'quat': [],
        'euler': [],
        'omega': [],
        'thrust': [],
        'torque': [],
        'phase': [],
        'target_pos': []
    }

    t = 0.0
    control_out = controller.compute_control(state, t=t, dt=dt_ctrl)

    # 4. Main Simulation Loop
    while t < max_time:
        euler = env.get_euler_angles()
        history['time'].append(t)
        history['pos'].append(state['p'].copy())
        history['vel'].append(state['v'].copy())
        history['quat'].append(state['q'].copy())
        history['euler'].append(euler.copy())
        history['omega'].append(state['omega'].copy())
        history['thrust'].append(control_out['thrust'])
        history['torque'].append(control_out['torque'].copy())
        history['phase'].append(control_out['phase'])
        history['target_pos'].append(control_out['target_pos'].copy())

        # Check touchdown criteria
        p = state['p']
        v = state['v']
        vel_tol = float(mission_cfg.get('landing_criteria', {}).get('vel_tolerance', 0.20))
        pos_tol = float(mission_cfg.get('landing_criteria', {}).get('pos_tolerance', 0.05))

        if control_out['phase'] == 3 and p[2] <= pos_tol and np.linalg.norm(v) <= vel_tol:
            print(f"\n[Touchdown Event] Precision soft landing achieved at t = {t:.3f} s (z = {p[2]:.4f} m, vz = {v[2]:.4f} m/s, ||v|| = {np.linalg.norm(v):.4f} m/s)")
            break

        # Controller Update Loop (every dt_ctrl)
        control_out = controller.compute_control(state, t=t, dt=dt_ctrl)

        # Multi-rate Physics Integration (Substepping)
        sim_done = False
        for _ in range(ctrl_substeps):
            state, sim_done, info = env.step(u=control_out['u_raw'], dt=dt_sim)
            if sim_done:
                break

        t += dt_ctrl
        if sim_done:
            print(f"\n[Ground Contact Event] Ground boundary reached at t = {t:.3f} s")
            break

    # Convert history lists to numpy arrays
    for key in history:
        history[key] = np.array(history[key])

    # 5. Compute Rigorous Performance Metrics
    metrics = calculate_flight_metrics(history, mission_cfg)

    print("\n" + "-" * 50)
    print("          SCENARIO 3 FLIGHT METRICS")
    print("-" * 50)
    print(f"Total Flight Time:     {metrics['flight_time']:.2f} s")
    print(f"Final Position:        [{metrics['final_x']:.3f}, {metrics['final_y']:.3f}, {metrics['final_z']:.3f}] m")
    print(f"Position Error:        {metrics['pos_error']:.4f} m  (Target: < 0.05 m)")
    print(f"Touchdown vz:          {metrics['touchdown_vz']:.4f} m/s (Soft Landing: < 0.3 m/s)")
    print(f"Final Tilt Angle:      {metrics['final_tilt_deg']:.2f} deg")
    print(f"Minimum Altitude:      {metrics['min_altitude']:.4f} m (Underground Check: >= 0.0 m)")
    print(f"Total Impulse:         {metrics['total_impulse']:.4f} N*s")
    print(f"Control Effort:        {metrics['control_effort']:.1f} N^2*s")
    print(f"Soft Landing Status:   {'SUCCESS' if metrics['soft_landing_success'] else 'FAILED'}")
    print("-" * 50)

    # 6. Automatic Plot Generation (Saved strictly in results/ folder)
    results_dir = os.path.join(root_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    plot_3d_path = os.path.join(results_dir, 'pid_baseline_3d.png')
    plot_telem_path = os.path.join(results_dir, 'pid_baseline_telemetry.png')

    plot_3d_trajectory(history, mission_cfg=mission_cfg, save_path=plot_3d_path, show=False)
    plot_flight_telemetry(history, drone_cfg=drone_cfg, mission_cfg=mission_cfg,
                          metrics=metrics, save_path=plot_telem_path, show=False)

    # 7. Acceptance Criteria Check (Safe Navigation & Precision Landing)
    pass_pos = metrics['pos_error'] <= 0.05
    pass_altitude = metrics['min_altitude'] >= -1e-4
    pass_soft_touchdown = abs(metrics['touchdown_vz']) <= 0.35

    all_passed = pass_pos and pass_altitude and pass_soft_touchdown

    print("\n" + "=" * 70)
    if all_passed:
        print("[PASS] SCENARIO 3: BASELINE PID SIMULATION VERIFIED SUCCESSFULLY!")
        print(f"  - No underground penetration: min_altitude = {metrics['min_altitude']:.4f} m >= 0.0 m")
        print(f"  - Precision landing achieved: position error = {metrics['pos_error']:.4f} m (< 0.05 m)")
        print(f"  - Safe touchdown velocity: vz = {metrics['touchdown_vz']:.4f} m/s")
        print(f"  - Flight time: {metrics['flight_time']:.2f} s")
    else:
        print("[FAIL] Some criteria were not satisfied. Review parameters.")
    print("=" * 70)

    # 8. Interactive Animation / Video Generation
    if animate or save_anim:
        anim_save_path = os.path.join(results_dir, 'pid_baseline_animation.mp4') if save_anim else None
        create_flight_animation(
            history=history,
            mission_cfg=mission_cfg,
            drone_cfg=drone_cfg,
            save_path=anim_save_path,
            show=animate,
            fps=fps,
            speed_multiplier=speed
        )

    return all_passed, metrics, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scenario 3: 3-Phase PID Baseline Simulation")
    parser.add_argument("--animate", action="store_true", help="Launch interactive real-time 3D animation window on macOS")
    parser.add_argument("--save-anim", action="store_true", help="Encode and save MP4 animation to results/pid_baseline_animation.mp4")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (e.g. 1.0, 1.5, 2.0)")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS for animation (default: 30)")

    args = parser.parse_args()
    run_baseline_pid_simulation(
        animate=args.animate,
        save_anim=args.save_anim,
        speed=args.speed,
        fps=args.fps
    )
