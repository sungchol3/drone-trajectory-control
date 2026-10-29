"""
Interactive Real-Time 6-DOF Drone Flight Animation & Telemetry Dashboard
Inertial Frame: ENU (East-North-Up)

Features:
  - Real-time 6-DOF rigid-body multirotor airframe visualization (+X forward heading, 4 rotor discs)
  - Dynamic attitude tilting & thrust vector quiver (reflects pitch/roll and commanded thrust)
  - Dynamic ground projection shadow and 3D landing pad rings
  - Multi-phase color-coded trajectory ribbon
  - Synchronized dual-pane telemetry dashboard with real-time time cursor
  - Native macOS interactive GUI playback (mouse 3D rotation, zoom, spacebar pause/resume)
  - Optional headless video export (MP4 via ffmpeg, GIF via pillow)

Author: Sungcheol Ha (KAIST Flight Dynamics & GNC Research Project)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from typing import Dict, Any, Optional, Tuple


class DroneAirframeVisual:
    """
    Constructs and transforms 3D quadrotor airframe geometry from Body-fixed frame to Inertial ENU frame.
    """

    def __init__(self, arm_length: float = 0.40, rotor_radius: float = 0.12):
        self.arm_length = arm_length
        self.rotor_radius = rotor_radius

        # Body-frame Arm line vertices
        # Arm X (Rotor 2 -> Center -> Rotor 1 [Front])
        self.arm_x_b = np.array([[-arm_length, 0.0, 0.0],
                                 [ arm_length, 0.0, 0.0]], dtype=np.float64)
        # Arm Y (Rotor 4 -> Center -> Rotor 3)
        self.arm_y_b = np.array([[0.0, -arm_length, 0.0],
                                 [0.0,  arm_length, 0.0]], dtype=np.float64)

        # Rotor positions in body frame
        self.rotor_pos_b = [
            np.array([ arm_length, 0.0, 0.0]),   # Rotor 1 (Front)
            np.array([-arm_length, 0.0, 0.0]),   # Rotor 2 (Rear)
            np.array([0.0,  arm_length, 0.0]),   # Rotor 3 (Left)
            np.array([0.0, -arm_length, 0.0])    # Rotor 4 (Right)
        ]

        # Rotor circle template in body frame (16 segments in X_B-Y_B plane)
        theta = np.linspace(0.0, 2.0 * np.pi, 16)
        self.rotor_circ_template = np.vstack([
            rotor_radius * np.cos(theta),
            rotor_radius * np.sin(theta),
            np.zeros_like(theta)
        ]).T  # Shape: (16, 3)

    @staticmethod
    def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
        """Converts unit quaternion [qw, qx, qy, qz] to 3x3 rotation matrix R_WB."""
        qw, qx, qy, qz = q
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-12:
            qw, qx, qy, qz = q / q_norm

        return np.array([
            [1.0 - 2.0*(qy*qy + qz*qz), 2.0*(qx*qy - qz*qw),     2.0*(qx*qz + qy*qw)],
            [2.0*(qx*qy + qz*qw),     1.0 - 2.0*(qx*qx + qz*qz), 2.0*(qy*qz - qx*qw)],
            [2.0*(qx*qz - qy*qw),     2.0*(qy*qz + qx*qw),     1.0 - 2.0*(qx*qx + qy*qy)]
        ], dtype=np.float64)

    def compute_world_geometry(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        thrust: float,
        mg: float = 304.0
    ) -> Dict[str, Any]:
        """
        Transforms all airframe elements into World (ENU) coordinates.
        """
        R_WB = self.quat_to_rot_matrix(quat)

        # 1. Transform Arms
        arm_x_w = (R_WB @ self.arm_x_b.T).T + pos
        arm_y_w = (R_WB @ self.arm_y_b.T).T + pos

        # 2. Transform Rotor Discs
        rotors_w = []
        for r_b in self.rotor_pos_b:
            circ_b = self.rotor_circ_template + r_b
            circ_w = (R_WB @ circ_b.T).T + pos
            rotors_w.append(circ_w)

        # 3. Thrust Vector (+Z_B in ENU frame)
        b3 = R_WB[:, 2]
        # Length scaled by thrust ratio: ~0.8m at hover
        arrow_len = float(np.clip(0.8 * (thrust / max(mg, 1.0)), 0.1, 1.8))
        thrust_end = pos + arrow_len * b3

        # 4. Ground Shadow (projected at z = 0)
        shadow_r = self.arm_length * 0.7
        shadow_theta = np.linspace(0.0, 2.0 * np.pi, 12)
        shadow_circ = np.vstack([
            pos[0] + shadow_r * np.cos(shadow_theta),
            pos[1] + shadow_r * np.sin(shadow_theta),
            np.zeros_like(shadow_theta)
        ]).T

        return {
            'pos': pos,
            'arm_x': arm_x_w,
            'arm_y': arm_y_w,
            'rotors': rotors_w,
            'thrust_base': pos,
            'thrust_end': thrust_end,
            'b3': b3,
            'shadow': shadow_circ
        }


class InteractiveFlightAnimator:
    """
    Dual-pane real-time flight animation and telemetry viewer.
    """

    def __init__(
        self,
        history: Dict[str, np.ndarray],
        mission_cfg: Optional[Dict[str, Any]] = None,
        drone_cfg: Optional[Dict[str, Any]] = None,
        speed_multiplier: float = 1.0,
        fps: int = 30
    ):
        self.history = history
        self.mission_cfg = mission_cfg or {}
        self.drone_cfg = drone_cfg or {}
        self.speed_multiplier = max(float(speed_multiplier), 0.1)
        self.fps = int(fps)

        # Vehicle and mission parameters
        self.mass = float(self.drone_cfg.get('mass', 31.0))
        self.gravity = float(self.drone_cfg.get('gravity', 9.80665))
        self.mg = self.mass * self.gravity
        geom = self.drone_cfg.get('geometry', {})
        arm_len = float(geom.get('arm_length', 0.40))
        self.visual_model = DroneAirframeVisual(arm_length=arm_len, rotor_radius=0.12)

        # Extract telemetry series
        self.time_arr = np.asarray(history['time'], dtype=np.float64)
        self.pos_arr = np.asarray(history['pos'], dtype=np.float64)
        self.vel_arr = np.asarray(history['vel'], dtype=np.float64)
        self.quat_arr = np.asarray(history.get('quat', np.tile([1, 0, 0, 0], (len(self.time_arr), 1))), dtype=np.float64)
        self.thrust_arr = np.asarray(history.get('thrust', np.full(len(self.time_arr), self.mg)), dtype=np.float64)
        self.phase_arr = np.asarray(history.get('phase', np.ones(len(self.time_arr), dtype=int)), dtype=int)
        self.euler_arr = history.get('euler', None)
        if self.euler_arr is None:
            # Fallback calculate pitch
            self.pitch_deg = np.zeros(len(self.time_arr))
        else:
            self.pitch_deg = np.rad2deg(self.euler_arr[:, 1])

        self.total_time = float(self.time_arr[-1])
        self.dt_data = float(np.mean(np.diff(self.time_arr))) if len(self.time_arr) > 1 else 0.02

        # Subsample frames to achieve exact 1.0x real-time playback at target fps
        sim_step_per_sec = 1.0 / self.dt_data
        stride = max(1, int(round(sim_step_per_sec / (self.fps * self.speed_multiplier))))
        self.frame_indices = np.arange(0, len(self.time_arr), stride)
        if self.frame_indices[-1] != len(self.time_arr) - 1:
            self.frame_indices = np.append(self.frame_indices, len(self.time_arr) - 1)

        self.n_frames = len(self.frame_indices)
        self.frame_interval_ms = int(1000.0 / self.fps)

        # Animation state controls
        self.is_paused = False
        self.anim = None

    def setup_figure(self) -> Tuple[plt.Figure, Any, Any]:
        """Sets up academic publication-grade dual-pane figure."""
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

        fig = plt.figure(figsize=(15, 8.5), dpi=120)
        fig.patch.set_facecolor('#F8F9FA')

        # Left 3D Flight View (Takes 55% width)
        ax_3d = fig.add_subplot(1, 2, 1, projection='3d')
        ax_3d.set_facecolor('#FFFFFF')

        # Right Telemetry Side Panel (3 stacked subplots taking 45% width)
        # Using GridSpec on right half
        import matplotlib.gridspec as gridspec
        gs_right = gridspec.GridSpec(3, 1, figure=fig, left=0.55, right=0.96,
                                     top=0.90, bottom=0.08, hspace=0.35)
        ax_alt = fig.add_subplot(gs_right[0, 0])
        ax_vel = fig.add_subplot(gs_right[1, 0])
        ax_thr = fig.add_subplot(gs_right[2, 0])

        return fig, ax_3d, (ax_alt, ax_vel, ax_thr)

    def init_3d_scene(self, ax_3d: Any) -> Dict[str, Any]:
        """Initializes all static elements and artists for the 3D scene."""
        pos = self.pos_arr
        x_min, x_max = min(np.min(pos[:, 0]), -11.0) - 1.0, max(np.max(pos[:, 0]), 1.0) + 1.5
        y_min, y_max = -2.5, 2.5
        z_min, z_max = 0.0, max(np.max(pos[:, 2]), 10.0) + 1.0

        ax_3d.set_xlim([x_min, x_max])
        ax_3d.set_ylim([y_min, y_max])
        ax_3d.set_zlim([z_min, z_max])
        ax_3d.set_xlabel('East [m]', labelpad=6, fontsize=9)
        ax_3d.set_ylabel('North [m]', labelpad=6, fontsize=9)
        ax_3d.set_zlabel('Altitude Up [m]', labelpad=6, fontsize=9)
        ax_3d.view_init(elev=22, azim=-65)

        # 1. Ground wireframe grid at z = 0
        gx = np.linspace(x_min, x_max, 12)
        gy = np.linspace(y_min, y_max, 8)
        GX, GY = np.meshgrid(gx, gy)
        GZ = np.zeros_like(GX)
        ax_3d.plot_wireframe(GX, GY, GZ, color='#E2E8F0', alpha=0.7, linewidth=0.6)

        # 2. Target Landing Pad at (0, 0, 0)
        target_pos = np.array([0.0, 0.0, 0.0])
        pad_theta = np.linspace(0, 2*np.pi, 60)
        for r_pad, col, ls, lw in [(0.5, '#E53E3E', '-', 2.0), (1.0, '#E53E3E', ':', 1.2), (1.8, '#CBD5E0', '--', 0.8)]:
            px = target_pos[0] + r_pad * np.cos(pad_theta)
            py = target_pos[1] + r_pad * np.sin(pad_theta)
            pz = np.zeros_like(pad_theta)
            ax_3d.plot(px, py, pz, color=col, linestyle=ls, linewidth=lw, alpha=0.85)

        # Start marker and target marker
        ax_3d.scatter([pos[0, 0]], [pos[0, 1]], [pos[0, 2]], color='#38A169', s=70, marker='D',
                      edgecolors='#1C4532', label=f'Start ({pos[0, 0]:.0f}, {pos[0, 2]:.0f}m)')
        ax_3d.scatter([target_pos[0]], [target_pos[1]], [target_pos[2]], color='#E53E3E', s=90, marker='X',
                      edgecolors='#742A2A', label='Landing Target (0, 0m)')

        # Phase Waypoints
        ax_3d.scatter([-10.0, 0.0], [0.0, 0.0], [5.0, 5.0], color='#DD6B20', s=45, marker='o',
                      edgecolors='black', label='3-Phase Waypoints (z=5m)')

        # Dynamic Artists for Drone
        # Front Arm (+X) in Crimson Red, Rear/Side Arms in Slate Gray
        arm_front_line, = ax_3d.plot([], [], [], color='#E53E3E', linewidth=3.5, label='Airframe (+X Front)')
        arm_cross_line, = ax_3d.plot([], [], [], color='#4A5568', linewidth=2.8)

        # 4 Rotor Discs
        rotor_lines = [ax_3d.plot([], [], [], color='#2B6CB0', linewidth=1.4)[0] for _ in range(4)]

        # Thrust vector line & tip
        thrust_line, = ax_3d.plot([], [], [], color='#D69E2E', linewidth=2.8, alpha=0.9, label='Thrust Vector')
        thrust_tip = ax_3d.scatter([], [], [], color='#D69E2E', s=35, marker='^')

        # Ground Shadow
        shadow_line, = ax_3d.plot([], [], [], color='#A0AEC0', linestyle='--', linewidth=1.2, alpha=0.8)

        # Trajectory Trail Ribbon (Color-coded)
        trail_p1, = ax_3d.plot([], [], [], color='#38A169', linewidth=2.0, alpha=0.85, label='Phase 1 (Descent)')
        trail_p2, = ax_3d.plot([], [], [], color='#DD6B20', linewidth=2.0, alpha=0.85, label='Phase 2 (Cruise)')
        trail_p3, = ax_3d.plot([], [], [], color='#3182CE', linewidth=2.2, alpha=0.9, label='Phase 3 (Precision Land)')

        # Real-time HUD Text overlay on 3D axes
        hud_text = ax_3d.text2D(0.03, 0.94, "", transform=ax_3d.transAxes, fontsize=9.5,
                                family='monospace', fontweight='bold',
                                bbox=dict(boxstyle='round,pad=0.5', facecolor='#1A202C', edgecolor='none', alpha=0.85),
                                color='#EDF2F7')

        ax_3d.legend(loc='lower left', fontsize=7.5, framealpha=0.8, facecolor='#FFFFFF')
        ax_3d.set_title("6-DOF Multirotor Real-Time Trajectory & Attitude Flight", fontsize=11, fontweight='bold', pad=10)

        return {
            'arm_front': arm_front_line,
            'arm_cross': arm_cross_line,
            'rotors': rotor_lines,
            'thrust_line': thrust_line,
            'thrust_tip': thrust_tip,
            'shadow': shadow_line,
            'trail_p1': trail_p1,
            'trail_p2': trail_p2,
            'trail_p3': trail_p3,
            'hud_text': hud_text
        }

    def init_telemetry_scene(self, axes: Tuple[Any, Any, Any]) -> Dict[str, Any]:
        """Plots reference telemetry curves and creates vertical time cursor lines."""
        ax_alt, ax_vel, ax_thr = axes
        t_arr = self.time_arr

        # 1. Altitude Subplot
        ax_alt.plot(t_arr, self.pos_arr[:, 2], color='#2B6CB0', linewidth=1.8, label='Altitude z(t)')
        ax_alt.axhline(5.0, color='#DD6B20', linestyle='--', linewidth=1.0, alpha=0.7, label='Cruise Alt (5m)')
        ax_alt.axhline(0.0, color='#E53E3E', linestyle='-', linewidth=1.0, alpha=0.7, label='Ground (0m)')
        ax_alt.set_ylabel('Altitude [m]', fontsize=8.5)
        ax_alt.set_xlim([0, self.total_time])
        ax_alt.set_ylim([-0.5, 11.0])
        ax_alt.grid(True, linestyle='--', alpha=0.5)
        ax_alt.legend(loc='upper right', fontsize=7.5, framealpha=0.8)
        cursor_alt = ax_alt.axvline(0.0, color='#E53E3E', linewidth=1.5, alpha=0.9)

        # 2. Velocity Subplot
        ax_vel.plot(t_arr, self.vel_arr[:, 0], color='#DD6B20', linewidth=1.5, label='vx (East)')
        ax_vel.plot(t_arr, self.vel_arr[:, 2], color='#38A169', linewidth=1.5, label='vz (Sink)')
        ax_vel.axhline(0.0, color='#A0AEC0', linestyle='--', linewidth=0.8)
        ax_vel.set_ylabel('Velocity [m/s]', fontsize=8.5)
        ax_vel.set_xlim([0, self.total_time])
        ax_vel.set_ylim([min(np.min(self.vel_arr[:, [0, 2]]), -3.5) - 0.5,
                         max(np.max(self.vel_arr[:, [0, 2]]), 4.0) + 0.5])
        ax_vel.grid(True, linestyle='--', alpha=0.5)
        ax_vel.legend(loc='upper right', fontsize=7.5, framealpha=0.8)
        cursor_vel = ax_vel.axvline(0.0, color='#E53E3E', linewidth=1.5, alpha=0.9)

        # 3. Thrust & Pitch Subplot
        line_thr, = ax_thr.plot(t_arr, self.thrust_arr, color='#D69E2E', linewidth=1.6, label='Thrust T(t) [N]')
        ax_thr.axhline(self.mg, color='#718096', linestyle='--', linewidth=1.0, alpha=0.8, label=f'Hover mg ({self.mg:.0f}N)')
        ax_thr.set_ylabel('Thrust [N]', fontsize=8.5, color='#B7791F')
        ax_thr.set_xlabel('Flight Time [s]', fontsize=9)
        ax_thr.set_xlim([0, self.total_time])
        ax_thr.set_ylim([150.0, max(np.max(self.thrust_arr), 450.0) + 30.0])
        ax_thr.grid(True, linestyle='--', alpha=0.5)

        # Twin axis for Pitch Angle
        ax_pitch = ax_thr.twinx()
        line_pitch, = ax_pitch.plot(t_arr, self.pitch_deg, color='#805AD5', linestyle='-.', linewidth=1.3, label='Pitch [deg]')
        ax_pitch.set_ylabel('Pitch [deg]', fontsize=8.5, color='#6B46C1')
        ax_pitch.set_ylim([-30.0, 30.0])

        # Combined legend
        ax_thr.legend([line_thr, line_pitch], ['Thrust [N]', 'Pitch [deg]'], loc='upper right', fontsize=7.5, framealpha=0.8)
        cursor_thr = ax_thr.axvline(0.0, color='#E53E3E', linewidth=1.5, alpha=0.9)

        return {
            'cursor_alt': cursor_alt,
            'cursor_vel': cursor_vel,
            'cursor_thr': cursor_thr
        }

    def run(self, save_path: Optional[str] = None, show: bool = True) -> None:
        """
        Launches the interactive real-time animation or encodes to file if save_path is provided.
        """
        fig, ax_3d, tel_axes = self.setup_figure()
        artists_3d = self.init_3d_scene(ax_3d)
        artists_tel = self.init_telemetry_scene(tel_axes)

        # Keyboard interaction callbacks
        def on_key(event):
            if event.key == ' ':
                self.is_paused = not self.is_paused
                status = "PAUSED" if self.is_paused else "RESUMED"
                fig.canvas.manager.set_window_title(f"6-DOF Drone Simulator [{status}] (Press Space to Toggle)")
            elif event.key in ['q', 'escape']:
                plt.close(fig)

        fig.canvas.mpl_connect('key_press_event', on_key)
        fig.canvas.manager.set_window_title("6-DOF Drone Real-Time Interactive Flight Viewer (Drag mouse to rotate)")

        # Precompute boolean masks for trajectory phase coloring
        phase1_mask = (self.phase_arr == 1)
        phase2_mask = (self.phase_arr == 2)
        phase3_mask = (self.phase_arr == 3)

        def update(frame_num: int):
            if self.is_paused:
                return []

            idx = self.frame_indices[frame_num]
            t_curr = self.time_arr[idx]
            pos_curr = self.pos_arr[idx]
            vel_curr = self.vel_arr[idx]
            quat_curr = self.quat_arr[idx]
            thrust_curr = self.thrust_arr[idx]
            phase_curr = self.phase_arr[idx]

            # 1. Update Airframe Geometry
            geo = self.visual_model.compute_world_geometry(pos_curr, quat_curr, thrust_curr, self.mg)

            # Front arm: from center to Rotor 1 (+X forward)
            arm_front_pts = np.vstack([geo['pos'], geo['arm_x'][1]])
            artists_3d['arm_front'].set_data(arm_front_pts[:, 0], arm_front_pts[:, 1])
            artists_3d['arm_front'].set_3d_properties(arm_front_pts[:, 2])

            # Cross arms (Rear + Left + Right)
            arm_cross_pts = np.vstack([
                geo['arm_x'][0], geo['pos'],
                geo['arm_y'][0], geo['pos'],
                geo['arm_y'][1]
            ])
            artists_3d['arm_cross'].set_data(arm_cross_pts[:, 0], arm_cross_pts[:, 1])
            artists_3d['arm_cross'].set_3d_properties(arm_cross_pts[:, 2])

            # 4 Rotor Discs
            for i in range(4):
                rc = geo['rotors'][i]
                artists_3d['rotors'][i].set_data(rc[:, 0], rc[:, 1])
                artists_3d['rotors'][i].set_3d_properties(rc[:, 2])

            # Thrust Vector Arrow
            t_pts = np.vstack([geo['thrust_base'], geo['thrust_end']])
            artists_3d['thrust_line'].set_data(t_pts[:, 0], t_pts[:, 1])
            artists_3d['thrust_line'].set_3d_properties(t_pts[:, 2])
            artists_3d['thrust_tip']._offsets3d = ([geo['thrust_end'][0]], [geo['thrust_end'][1]], [geo['thrust_end'][2]])

            # Ground Shadow
            sh = geo['shadow']
            artists_3d['shadow'].set_data(sh[:, 0], sh[:, 1])
            artists_3d['shadow'].set_3d_properties(sh[:, 2])

            # 2. Update Trajectory Ribbons up to idx
            m1 = phase1_mask[:idx+1]
            m2 = phase2_mask[:idx+1]
            m3 = phase3_mask[:idx+1]

            if np.any(m1):
                p1_pts = self.pos_arr[:idx+1][m1]
                artists_3d['trail_p1'].set_data(p1_pts[:, 0], p1_pts[:, 1])
                artists_3d['trail_p1'].set_3d_properties(p1_pts[:, 2])
            if np.any(m2):
                p2_pts = self.pos_arr[:idx+1][m2]
                artists_3d['trail_p2'].set_data(p2_pts[:, 0], p2_pts[:, 1])
                artists_3d['trail_p2'].set_3d_properties(p2_pts[:, 2])
            if np.any(m3):
                p3_pts = self.pos_arr[:idx+1][m3]
                artists_3d['trail_p3'].set_data(p3_pts[:, 0], p3_pts[:, 1])
                artists_3d['trail_p3'].set_3d_properties(p3_pts[:, 2])

            # 3. Update Real-time HUD Text
            speed = np.linalg.norm(vel_curr)
            phase_names = {1: "Phase 1: Vertical Descent",
                           2: "Phase 2: Horizontal Cruise",
                           3: "Phase 3: Precision Flare Landing"}
            phase_str = phase_names.get(phase_curr, f"Phase {phase_curr}")
            hud_str = (
                f"TIME : {t_curr:6.2f} s / {self.total_time:.2f} s\n"
                f"MODE : {phase_str}\n"
                f"POS  : [{pos_curr[0]:+5.2f}, {pos_curr[1]:+5.2f}, {pos_curr[2]:5.2f}] m\n"
                f"SPEED: {speed:5.2f} m/s (vz: {vel_curr[2]:+5.2f} m/s)\n"
                f"THR  : {thrust_curr:5.1f} N | PITCH: {self.pitch_deg[idx]:+4.1f} deg"
            )
            artists_3d['hud_text'].set_text(hud_str)

            # 4. Update Telemetry Time Cursors
            artists_tel['cursor_alt'].set_xdata([t_curr, t_curr])
            artists_tel['cursor_vel'].set_xdata([t_curr, t_curr])
            artists_tel['cursor_thr'].set_xdata([t_curr, t_curr])

            return []

        # Create animation
        self.anim = FuncAnimation(
            fig, update,
            frames=self.n_frames,
            interval=self.frame_interval_ms,
            blit=False,
            repeat=True
        )

        # Handle saving if requested
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            print(f"\n[Animation] Encoding {self.n_frames} frames to video: {save_path} ...")
            if save_path.endswith('.mp4'):
                from matplotlib.animation import FFMpegWriter
                writer = FFMpegWriter(fps=self.fps, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])
                self.anim.save(save_path, writer=writer)
            elif save_path.endswith('.gif'):
                from matplotlib.animation import PillowWriter
                writer = PillowWriter(fps=min(self.fps, 20))
                self.anim.save(save_path, writer=writer)
            print(f"[Animation] Video successfully saved to: {save_path}")

        # Show interactive GUI window if requested
        if show:
            print("\n[Interactive Viewer] Launching native macOS GUI window.")
            print("  - Drag with mouse to rotate 3D view")
            print("  - Scroll / pinch to zoom")
            print("  - Press [Space] to pause/resume, [Q] to close")
            plt.show()
        else:
            plt.close(fig)


def create_flight_animation(
    history: Dict[str, np.ndarray],
    mission_cfg: Optional[Dict[str, Any]] = None,
    drone_cfg: Optional[Dict[str, Any]] = None,
    save_path: Optional[str] = None,
    show: bool = True,
    fps: int = 30,
    speed_multiplier: float = 1.0
) -> None:
    """
    Convenience wrapper to run InteractiveFlightAnimator.
    """
    animator = InteractiveFlightAnimator(
        history=history,
        mission_cfg=mission_cfg,
        drone_cfg=drone_cfg,
        speed_multiplier=speed_multiplier,
        fps=fps
    )
    animator.run(save_path=save_path, show=show)
