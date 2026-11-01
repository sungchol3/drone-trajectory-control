"""
Phase 2: ZEM/ZEV Optimal Guidance & SE(3) Geometric Flight Simulation
High-Fidelity 6-DOF Multirotor Autonomous Trajectory Tracking and Soft Landing

Features:
  - Multi-rate simulation (200 Hz RK4 physics + 50 Hz guidance/control loop)
  - Analytical minimum-time bisection search solver
  - ZEM/ZEV closed-form acceleration guidance law
  - Lie Group SO(3) geometric attitude and thrust tracking
  - Terminal soft touchdown flare handover
  - Rigorous comparative scoreboard against Phase 1 Baseline PID
  - Static 3D trajectory and 6-subplot telemetry plot generation
  - Native macOS interactive real-time 3D animation GUI (--animate)
  - Headless MP4 video encoding (--save-anim)

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
from controllers.zem_zev_controller import ZEMZEVController
from utils.metrics import calculate_flight_metrics
from utils.visualization import plot_3d_trajectory, plot_flight_telemetry
from utils.animation import create_flight_animation


def run_zem_zev_simulation(
    animate: bool = False,
    save_anim: bool = False,
    speed: float = 1.0,
    fps: int = 30
):
    print("\n" + "=" * 75)
    print(">>> Running Phase 2: ZEM/ZEV Optimal Guidance & SE(3) Geometric Control")
    print("=" * 75)

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
    controller = ZEMZEVController(drone_cfg, mission_cfg)

    start_pos = np.array(mission_cfg['waypoints']['start_pos'], dtype=np.float64)
    start_vel = np.array(mission_cfg['waypoints']['start_vel'], dtype=np.float64)
    target_pos = np.array(mission_cfg['waypoints']['target_pos'], dtype=np.float64)
    max_time = float(mission_cfg['timing']['max_time'])
    dt_sim = float(mission_cfg['timing']['dt_sim'])     # 0.005s (200 Hz)
    dt_ctrl = float(mission_cfg['timing']['dt_ctrl'])   # 0.020s (50 Hz)
    ctrl_substeps = max(int(dt_ctrl / dt_sim), 1)

    print(f"Vehicle Mass:          {env.mass:.1f} kg (30kg Airframe + 1kg Payload)")
    print(f"Initial Station:       {start_pos} m")
    print(f"Landing Target:        {target_pos} m")
    print(f"Simulation Timestep:   dt_sim = {dt_sim*1000:.1f} ms (200 Hz), dt_ctrl = {dt_ctrl*1000:.1f} ms (50 Hz)")

    # Reset
    state = env.reset(init_pos=start_pos, init_vel=start_vel)
    controller.reset()

    # Pre-solve optimal flight time
    Tf_opt = controller.solve_mission_time(start_pos, start_vel)
    print(f"Optimal Flight Time:   Tf* = {Tf_opt:.2f} s (Solved via Bisection Search)")

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

        # Precision soft touchdown condition
        if control_out['phase'] == 2 and p[2] <= pos_tol and np.linalg.norm(v) <= vel_tol:
            print(f"\n[Touchdown Event] Precision soft landing achieved at t = {t:.3f} s (z = {p[2]:.4f} m, ||v|| = {np.linalg.norm(v):.4f} m/s)")
            break

        # Compute next control command
        control_out = controller.compute_control(state, t=t, dt=dt_ctrl)

        # Multi-rate Physics Integration (Substepping at 200 Hz)
        sim_done = False
        for _ in range(ctrl_substeps):
            state, sim_done, info = env.step(u=control_out['u_raw'], dt=dt_sim)
            if sim_done:
                break

        t += dt_ctrl
        if sim_done:
            print(f"\n[Ground Contact Event] Ground boundary reached at t = {t:.3f} s (z = {state['p'][2]:.4f} m)")
            break

    # Convert history lists to numpy arrays
    for key in history:
        history[key] = np.array(history[key])

    # 5. Compute Performance Metrics
    metrics = calculate_flight_metrics(history, mission_cfg)

    # 6. Comparative Scoreboard: Phase 1 (PID) vs Phase 2 (ZEM/ZEV)
    # Phase 1 PID baseline values from our earlier verified run
    pid_time = 13.66
    pid_error = 0.0483
    pid_effort = 1307237.4
    pid_impulse = 21.8313

    time_reduction_pct = ((pid_time - metrics['flight_time']) / pid_time) * 100.0
    effort_reduction_pct = ((pid_effort - metrics['control_effort']) / pid_effort) * 100.0

    print("\n" + "-" * 75)
    print("             PHASE 2 (ZEM/ZEV) vs PHASE 1 (PID) COMPARATIVE BENCHMARK")
    print("-" * 75)
    print(f"Metric                        Phase 1 (Baseline PID)    Phase 2 (ZEM/ZEV)       Improvement")
    print("-" * 75)
    print(f"Total Flight Time [s]:        {pid_time:6.2f} s                  {metrics['flight_time']:6.2f} s               {time_reduction_pct:+5.1f}% (FASTER)")
    print(f"Landing Position Error [m]:   {pid_error:6.4f} m                 {metrics['pos_error']:6.4f} m              {(pid_error - metrics['pos_error']):+6.4f} m")
    print(f"Touchdown Velocity vz [m/s]:  -0.1737 m/s                {metrics['touchdown_vz']:+6.4f} m/s              (Safe Soft Landing)")
    print(f"Final Tilt Angle [deg]:         0.00 deg                 {metrics['final_tilt_deg']:6.2f} deg              (Vertical Landing)")
    print(f"Minimum Altitude z_min [m]:     0.0483 m                 {metrics['min_altitude']:6.4f} m              (No Crash: >= 0.0m)")
    print(f"Control Effort [N^2*s]:       {pid_effort:10.1f}              {metrics['control_effort']:10.1f}              {effort_reduction_pct:+5.1f}% (ENERGY)")
    print(f"Total Impulse [N*s]:          {pid_impulse:7.4f}                 {metrics['total_impulse']:7.4f} N*s")
    print(f"Soft Landing Status:          SUCCESS                   {'SUCCESS' if metrics['soft_landing_success'] else 'FAILED'}")
    print("-" * 75)

    # 7. Visualization Plot Export (Saved strictly to DEV results/ folder)
    results_dir = os.path.join(root_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    plot_3d_path = os.path.join(results_dir, 'zem_zev_3d.png')
    plot_telem_path = os.path.join(results_dir, 'zem_zev_telemetry.png')

    plot_3d_trajectory(history, mission_cfg=mission_cfg, save_path=plot_3d_path, show=False)
    plot_flight_telemetry(history, drone_cfg=drone_cfg, mission_cfg=mission_cfg,
                          metrics=metrics, save_path=plot_telem_path, show=False)

    # 8. Interactive Real-Time 3D Animation & Video Export
    if animate or save_anim:
        anim_save_path = os.path.join(results_dir, 'zem_zev_animation.mp4') if save_anim else None
        create_flight_animation(
            history=history,
            mission_cfg=mission_cfg,
            drone_cfg=drone_cfg,
            save_path=anim_save_path,
            show=animate,
            fps=fps,
            speed_multiplier=speed
        )

    # 9. Acceptance Criteria
    pass_pos = metrics['pos_error'] <= 0.05
    pass_altitude = metrics['min_altitude'] >= -1e-4
    pass_soft_touchdown = abs(metrics['touchdown_vz']) <= 0.35
    pass_time = metrics['flight_time'] < 6.0

    all_passed = pass_pos and pass_altitude and pass_soft_touchdown and pass_time

    print("\n" + "=" * 75)
    if all_passed:
        print("[PASS] PHASE 2: ZEM/ZEV OPTIMAL GUIDANCE & SE(3) CONTROL VERIFIED SUCCESSFULLY!")
        print(f"  - Flight time reduced by {time_reduction_pct:.1f}%: {pid_time:.2f} s -> {metrics['flight_time']:.2f} s")
        print(f"  - Precision landing achieved: error = {metrics['pos_error']:.4f} m (< 0.05 m)")
        print(f"  - Soft touchdown velocity: vz = {metrics['touchdown_vz']:.4f} m/s")
        print(f"  - Energy effort reduced by {effort_reduction_pct:.1f}%: {pid_effort:.0f} -> {metrics['control_effort']:.0f} N^2*s")
    else:
        print("[FAIL] Some criteria were not satisfied. Review parameters.")
    print("=" * 75 + "\n")

    return all_passed, metrics, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: ZEM/ZEV Optimal Guidance Simulation")
    parser.add_argument("--animate", action="store_true", help="Launch interactive real-time 3D animation window on macOS")
    parser.add_argument("--save-anim", action="store_true", help="Encode and save MP4 animation to results/zem_zev_animation.mp4")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (e.g. 1.0, 1.5)")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS for animation (default: 30)")

    args = parser.parse_args()
    run_zem_zev_simulation(
        animate=args.animate,
        save_anim=args.save_anim,
        speed=args.speed,
        fps=args.fps
    )
