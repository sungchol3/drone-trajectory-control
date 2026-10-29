"""
High-Fidelity 6-DOF Drone Trajectory and Flight Telemetry Visualization
Inertial Frame: ENU (East-North-Up)

Author: Sungcheol Ha (KAIST Flight Dynamics & GNC Research Project)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Any, Optional


def set_academic_style():
    """Sets a clean, publication-ready academic plotting style."""
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['grid.color'] = '#E0E0E0'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.5


def plot_3d_trajectory(
    history: Dict[str, np.ndarray],
    mission_cfg: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    quiver_interval: float = 0.5
) -> plt.Figure:
    """
    Renders 3D trajectory with attitude thrust vectors, ground pad, and key mission waypoints.
    """
    set_academic_style()
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    pos = history['pos']
    time = history['time']
    quat = history.get('quat', None)

    # 1. Plot ground plane grid at z = 0
    x_min, x_max = min(np.min(pos[:, 0]), -12.0) - 1.0, max(np.max(pos[:, 0]), 2.0) + 1.0
    y_min, y_max = min(np.min(pos[:, 1]), -2.0) - 1.0, max(np.max(pos[:, 1]), 2.0) + 1.0
    gx = np.linspace(x_min, x_max, 10)
    gy = np.linspace(y_min, y_max, 10)
    GX, GY = np.meshgrid(gx, gy)
    GZ = np.zeros_like(GX)
    ax.plot_wireframe(GX, GY, GZ, color='#CCCCCC', alpha=0.4, linewidth=0.6)

    # 2. Plot Landing Target Pad at z = 0
    target_pos = np.array([0.0, 0.0, 0.0])
    if mission_cfg is not None:
        target_pos = np.array(mission_cfg.get('waypoints', {}).get('target_pos', [0.0, 0.0, 0.0]))
    
    pad_theta = np.linspace(0, 2 * np.pi, 50)
    pad_r1 = 0.5
    pad_r2 = 1.0
    ax.plot(target_pos[0] + pad_r1 * np.cos(pad_theta), 
            target_pos[1] + pad_r1 * np.sin(pad_theta), 
            np.zeros_like(pad_theta), color='#E74C3C', linewidth=1.5, label='Landing Pad (r=0.5m)')
    ax.plot(target_pos[0] + pad_r2 * np.cos(pad_theta), 
            target_pos[1] + pad_r2 * np.sin(pad_theta), 
            np.zeros_like(pad_theta), color='#E74C3C', linestyle=':', linewidth=1.0)

    # 3. 3D Flight Trajectory Line (Color-coded by flight time)
    sc = ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c=time, cmap='viridis', s=2, alpha=0.8)
    cbar = fig.colorbar(sc, ax=ax, pad=0.08, shrink=0.6)
    cbar.set_label('Flight Time [s]', fontsize=9)

    # 4. Start, Waypoints, and Target Markers
    start_pos = pos[0]
    ax.scatter([start_pos[0]], [start_pos[1]], [start_pos[2]], color='#27AE60', s=80, marker='D', 
               edgecolors='black', label=f'Start ({start_pos[0]:.0f}, {start_pos[1]:.0f}, {start_pos[2]:.0f})m', zorder=10)
    
    # 3-Stage Heuristic Waypoints marker
    wp1 = np.array([-10.0, 0.0, 5.0])
    wp2 = np.array([0.0, 0.0, 5.0])
    ax.scatter([wp1[0], wp2[0]], [wp1[1], wp2[1]], [wp1[2], wp2[2]], color='#F39C12', s=50, marker='o', 
               edgecolors='black', label='3-Phase Waypoints (z=5m)', zorder=9)

    ax.scatter([target_pos[0]], [target_pos[1]], [target_pos[2]], color='#C0392B', s=100, marker='X', 
               edgecolors='black', label=f'Target ({target_pos[0]:.0f}, {target_pos[1]:.0f}, {target_pos[2]:.0f})m', zorder=10)

    # 5. Body Attitude Quivers (+Z_B thrust axis in Inertial Frame)
    if quat is not None:
        dt = np.mean(np.diff(time)) if len(time) > 1 else 0.005
        step_interval = max(int(quiver_interval / dt), 1)
        
        quiver_indices = np.arange(0, len(time), step_interval)
        qx_pos = pos[quiver_indices, 0]
        qy_pos = pos[quiver_indices, 1]
        qz_pos = pos[quiver_indices, 2]
        
        # Calculate R_WB[:, 2] for each index
        b3_x, b3_y, b3_z = [], [], []
        scale = 0.8  # vector length
        for idx in quiver_indices:
            qw, qx, qy, qz = quat[idx]
            # R_WB[:, 2] = [2*(qx*qz + qy*qw), 2*(qy*qz - qx*qw), 1 - 2*(qx*qx + qy*qy)]
            r13 = 2.0 * (qx * qz + qy * qw)
            r23 = 2.0 * (qy * qz - qx * qw)
            r33 = 1.0 - 2.0 * (qx * qx + qy * qy)
            b3_x.append(scale * r13)
            b3_y.append(scale * r23)
            b3_z.append(scale * r33)

        ax.quiver(qx_pos, qy_pos, qz_pos, b3_x, b3_y, b3_z, color='#2980B9', alpha=0.7, 
                  linewidth=1.0, length=1.0, arrow_length_ratio=0.3, label='Body Thrust Axis (+Z_B)')

    # Labels and View Angle
    ax.set_xlabel('East X [m]', fontsize=10, labelpad=8)
    ax.set_ylabel('North Y [m]', fontsize=10, labelpad=8)
    ax.set_zlabel('Altitude Z [m]', fontsize=10, labelpad=8)
    ax.set_title('3D Multirotor Trajectory & Attitude Quiver (ENU Frame)', fontsize=12, fontweight='bold', pad=15)
    ax.view_init(elev=24, azim=-55)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Visualization] 3D trajectory plot saved to: {save_path}")

    if show:
        plt.show()

    return fig


def plot_flight_telemetry(
    history: Dict[str, np.ndarray],
    drone_cfg: Optional[Dict[str, Any]] = None,
    mission_cfg: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None,
    show: bool = False
) -> plt.Figure:
    """
    Renders 6-subplot flight telemetry time-series with FSM phase shading and performance scoreboard.
    """
    set_academic_style()
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), dpi=150, sharex=True)

    time = history['time']
    pos = history['pos']
    vel = history['vel']
    euler = history.get('euler', np.zeros_like(pos))
    omega = history.get('omega', np.zeros_like(pos))
    thrust = history.get('thrust', np.zeros(len(time)))
    torque = history.get('torque', np.zeros_like(pos))
    phase = history.get('phase', None)
    target_pos = history.get('target_pos', None)

    # Physical limits from config
    mass = 31.0
    gravity = 9.80665
    thrust_max = 600.0
    tilt_max_deg = 35.0
    if drone_cfg is not None:
        mass = float(drone_cfg.get('mass', 31.0))
        gravity = float(drone_cfg.get('gravity', 9.80665))
        thrust_max = float(drone_cfg.get('geometry', {}).get('thrust_max', 600.0))
        tilt_max_deg = float(drone_cfg.get('geometry', {}).get('tilt_max_deg', 35.0))
    hover_thrust = mass * gravity

    # Helper function to add FSM Phase shading across all subplots
    def add_phase_shading(ax):
        if phase is None or len(phase) != len(time):
            return
        # Find phase transitions
        diffs = np.diff(phase)
        transition_indices = np.where(diffs != 0)[0]
        boundaries = [0] + list(transition_indices + 1) + [len(time) - 1]

        colors = {1: '#E8F8F5', 2: '#FEF9E7', 3: '#EAFAF1'}  # Phase 1: Cyan, 2: Yellow, 3: Mint
        labels = {1: 'Phase 1: Descent', 2: 'Phase 2: XY Align', 3: 'Phase 3: Landing'}

        added_labels = set()
        for i in range(len(boundaries) - 1):
            idx_start = boundaries[i]
            idx_end = boundaries[i + 1]
            p_val = int(phase[idx_start])
            c = colors.get(p_val, '#F2F3F4')
            lbl = labels.get(p_val, f'Phase {p_val}') if p_val not in added_labels else None
            added_labels.add(p_val)
            ax.axvspan(time[idx_start], time[idx_end], color=c, alpha=0.5, zorder=0, label=lbl)

    # ---------------- (1, 1) Position Tracking ----------------
    ax = axes[0, 0]
    add_phase_shading(ax)
    ax.plot(time, pos[:, 0], color='#2980B9', linewidth=1.5, label='x (East)')
    ax.plot(time, pos[:, 1], color='#8E44AD', linewidth=1.5, label='y (North)')
    ax.plot(time, pos[:, 2], color='#27AE60', linewidth=1.8, label='z (Altitude)')
    if target_pos is not None:
        ax.plot(time, target_pos[:, 0], color='#2980B9', linestyle=':', alpha=0.7, label='x_ref')
        ax.plot(time, target_pos[:, 1], color='#8E44AD', linestyle=':', alpha=0.7, label='y_ref')
        ax.plot(time, target_pos[:, 2], color='#27AE60', linestyle=':', alpha=0.7, label='z_ref')
    ax.axhline(0.0, color='#C0392B', linestyle='--', linewidth=1.0, label='Ground (z=0)')
    ax.set_ylabel('Position [m]', fontsize=9, fontweight='bold')
    ax.grid(True)
    ax.legend(loc='upper right', fontsize=7, ncol=3, framealpha=0.9)
    ax.set_title('(a) 3D Inertial Position Tracking', fontsize=10, fontweight='bold')

    # ---------------- (1, 2) Linear Velocities ----------------
    ax = axes[0, 2 - 1]
    add_phase_shading(ax)
    ax.plot(time, vel[:, 0], color='#2980B9', linewidth=1.2, label='vx')
    ax.plot(time, vel[:, 1], color='#8E44AD', linewidth=1.2, label='vy')
    ax.plot(time, vel[:, 2], color='#27AE60', linewidth=1.5, label='vz (Climb/Descent)')
    ax.axhline(0.0, color='black', linestyle=':', linewidth=0.8)
    ax.set_ylabel('Velocity [m/s]', fontsize=9, fontweight='bold')
    ax.grid(True)
    ax.legend(loc='upper right', fontsize=8, ncol=3, framealpha=0.9)
    ax.set_title('(b) Inertial Linear Velocity', fontsize=10, fontweight='bold')

    # ---------------- (2, 1) Euler Attitude Angles ----------------
    ax = axes[1, 0]
    add_phase_shading(ax)
    roll_deg = np.rad2deg(euler[:, 0])
    pitch_deg = np.rad2deg(euler[:, 1])
    yaw_deg = np.rad2deg(euler[:, 2])
    ax.plot(time, roll_deg, color='#D35400', linewidth=1.2, label='Roll phi')
    ax.plot(time, pitch_deg, color='#2980B9', linewidth=1.2, label='Pitch theta')
    ax.plot(time, yaw_deg, color='#7F8C8D', linewidth=1.0, linestyle='-.', label='Yaw psi')
    ax.axhline(tilt_max_deg, color='#C0392B', linestyle='--', linewidth=0.8, label=f'+/-{tilt_max_deg:.0f} deg Limit')
    ax.axhline(-tilt_max_deg, color='#C0392B', linestyle='--', linewidth=0.8)
    ax.set_ylabel('Attitude [deg]', fontsize=9, fontweight='bold')
    ax.grid(True)
    ax.legend(loc='upper right', fontsize=8, ncol=3, framealpha=0.9)
    ax.set_title('(c) Euler Angles (Roll / Pitch / Yaw)', fontsize=10, fontweight='bold')

    # ---------------- (2, 2) Body Angular Rates ----------------
    ax = axes[1, 1]
    add_phase_shading(ax)
    p_deg = np.rad2deg(omega[:, 0])
    q_deg = np.rad2deg(omega[:, 1])
    r_deg = np.rad2deg(omega[:, 2])
    ax.plot(time, p_deg, color='#D35400', linewidth=1.0, label='p (Roll rate)')
    ax.plot(time, q_deg, color='#2980B9', linewidth=1.0, label='q (Pitch rate)')
    ax.plot(time, r_deg, color='#7F8C8D', linewidth=1.0, label='r (Yaw rate)')
    ax.axhline(0.0, color='black', linestyle=':', linewidth=0.8)
    ax.set_ylabel('Body Rates [deg/s]', fontsize=9, fontweight='bold')
    ax.grid(True)
    ax.legend(loc='upper right', fontsize=8, ncol=3, framealpha=0.9)
    ax.set_title('(d) Body Angular Velocity', fontsize=10, fontweight='bold')

    # ---------------- (3, 1) Total Collective Thrust ----------------
    ax = axes[2, 0]
    add_phase_shading(ax)
    ax.plot(time, thrust, color='#16A085', linewidth=1.5, label='Thrust T(t)')
    ax.axhline(hover_thrust, color='#8E44AD', linestyle=':', linewidth=1.2, 
               label=f'Hover T_hover={hover_thrust:.1f} N')
    ax.axhline(thrust_max, color='#C0392B', linestyle='--', linewidth=1.0, 
               label=f'Max Limit T_max={thrust_max:.0f} N')
    ax.set_xlabel('Time [s]', fontsize=10, fontweight='bold')
    ax.set_ylabel('Thrust [N]', fontsize=9, fontweight='bold')
    ax.grid(True)
    ax.legend(loc='upper right', fontsize=8, ncol=3, framealpha=0.9)
    ax.set_title('(e) Collective Actuator Thrust', fontsize=10, fontweight='bold')

    # ---------------- (3, 2) Control Torques ----------------
    ax = axes[2, 1]
    add_phase_shading(ax)
    ax.plot(time, torque[:, 0], color='#D35400', linewidth=1.0, label='tau_x (Roll)')
    ax.plot(time, torque[:, 1], color='#2980B9', linewidth=1.0, label='tau_y (Pitch)')
    ax.plot(time, torque[:, 2], color='#7F8C8D', linewidth=1.0, label='tau_z (Yaw)')
    ax.axhline(0.0, color='black', linestyle=':', linewidth=0.8)
    ax.set_xlabel('Time [s]', fontsize=10, fontweight='bold')
    ax.set_ylabel('Torque [N*m]', fontsize=9, fontweight='bold')
    ax.grid(True)
    ax.legend(loc='upper right', fontsize=8, ncol=3, framealpha=0.9)
    ax.set_title('(f) 3-Axis Body Control Torques', fontsize=10, fontweight='bold')

    # ---------------- Scoreboard Overlay ----------------
    if metrics is not None:
        status_txt = "SUCCESS (Soft Landing)" if metrics.get('soft_landing_success', False) else "FAILED (Landing Out of Bounds)"
        score_txt = (
            f"Flight Time: {metrics.get('flight_time', 0.0):.2f} s | "
            f"Pos Error: {metrics.get('pos_error', 0.0):.3f} m | "
            f"Touchdown vz: {metrics.get('touchdown_vz', 0.0):.3f} m/s | "
            f"Tilt: {metrics.get('final_tilt_deg', 0.0):.1f}° | "
            f"Impulse: {metrics.get('total_impulse', 0.0):.3f} N*s | "
            f"Status: {status_txt}"
        )
        fig.suptitle(f"6-DOF Drone Flight Telemetry & Performance Scoreboard\n[{score_txt}]", 
                     fontsize=11, fontweight='bold', y=0.995)
    else:
        fig.suptitle("6-DOF Drone Flight Telemetry and Dynamic States", fontsize=12, fontweight='bold', y=0.99)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[Visualization] Telemetry plot saved to: {save_path}")

    if show:
        plt.show()

    return fig
