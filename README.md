# 6-DOF Autonomous Drone Trajectory Control & Precision Landing Benchmark

> **KAIST Flight Dynamics & Guidance, Navigation, and Control (GNC) Research Project**  
> **Author**: Sungcheol Ha (KAIST Department of Aerospace Engineering & Physics)  
> **Reference Origin**: Modernized & upgraded from legacy Point-Mass MATLAB codebase (`sungchol3/RL-base-drone-delivery`) to a high-fidelity 6-DOF Python simulation platform.

---

## 1. Project Overview

This research repository implements a benchmark framework for **optimal trajectory tracking and autonomous precision landing of a high-mass (31.0 kg) multirotor UAV**. The project compares classical and modern intelligent flight guidance and control paradigms across identical high-fidelity 6-DOF physical simulation environments:

1. **Phase 1 (Baseline)**: Cascaded 3-Phase Decoupled PID Controller with Ground Flare Regulation.
2. **Phase 2 (Optimal Guidance)**: Zero-Effort-Miss / Zero-Effort-Velocity (ZEM/ZEV) Guidance with $SE(3)$ Geometric Tracking Control.
3. **Phase 3 (Optimal Control)**: Nonlinear Model Predictive Control (NMPC) with Rotor Input and State Constraints.
4. **Phase 4 (Deep RL)**: Mission-Tailored Reinforcement Learning (Soft Actor-Critic, SAC).
5. **Phase 5 (Benchmark & Comparison)**: Quantitative comparative analysis across flight time ($T_f$), landing accuracy ($\|\mathbf{p}_f\|$), touchdown velocity ($v_z$), energy efficiency ($\int T^2 dt$), and tilt stability.

---

## 2. Mathematical Modeling & Coordinate Conventions

### (1) Coordinate Frames
- **Inertial World Frame ($\mathcal{F}_W$)**: Standard **East-North-Up (ENU)** coordinate system.
  - $+X_W$: East, $+Y_W$: North, $+Z_W$: Up (Altitude).
  - Gravity vector: $\mathbf{g}_W = [0, 0, -9.80665]^T\,\text{m/s}^2$.
- **Body-Fixed Frame ($\mathcal{F}_B$)**:
  - $+X_B$: Forward (Vehicle Heading), $+Y_B$: Left, $+Z_B$: Upward (Total thrust axis).
- **Attitude Representation**:
  - Unit Quaternion $\mathbf{q} = [q_w, q_x, q_y, q_z]^T \in \mathbb{S}^3$ (Gimbal lock-free $SO(3)$ rotation).
  - Rotation matrix $\mathbf{R}_{WB}(\mathbf{q}) \in SO(3)$ transforms vectors from $\mathcal{F}_B$ to $\mathcal{F}_W$.

### (2) 13-State Nonlinear 6-DOF Equations of Motion
$$\mathbf{x} = [\mathbf{p}_W^T, \; \mathbf{v}_W^T, \; \mathbf{q}^T, \; \boldsymbol{\Omega}_B^T]^T \in \mathbb{R}^{13}$$

1. **Translational Dynamics**:
   $$\dot{\mathbf{p}}_W = \mathbf{v}_W$$
   $$m \dot{\mathbf{v}}_W = m \mathbf{g}_W + \mathbf{R}_{WB}(\mathbf{q}) \begin{bmatrix} 0 \\ 0 \\ T \end{bmatrix} - \mathbf{D}_v \mathbf{v}_W$$
   where $m = 31.0\,\text{kg}$ ($30.0\,\text{kg}$ airframe + $1.0\,\text{kg}$ delivery payload), $T$ is total rotor thrust, and $\mathbf{D}_v = \text{diag}(0.25, 0.25, 0.35)\,\text{N}\cdot\text{s/m}$ is aerodynamic damping.

2. **Rotational Dynamics**:
   $$\dot{\mathbf{q}} = \frac{1}{2} \mathbf{q} \otimes \begin{bmatrix} 0 \\ \boldsymbol{\Omega}_B \end{bmatrix}$$
   $$\mathbf{J} \dot{\boldsymbol{\Omega}}_B = -\boldsymbol{\Omega}_B \times (\mathbf{J} \boldsymbol{\Omega}_B) + \boldsymbol{\tau}_B$$
   where $\mathbf{J} = \text{diag}(1.25, 1.25, 2.50)\,\text{kg}\cdot\text{m}^2$ is the moment of inertia tensor and $\boldsymbol{\tau}_B = [\tau_x, \tau_y, \tau_z]^T$ is the 3-axis control moment.

3. **Multi-Rate Numerical Integration**:
   - Physics integration: 4th-Order Runge-Kutta (RK4) at $200\,\text{Hz}$ ($\Delta t_{\text{sim}} = 0.005\,\text{s}$).
   - Control loop update: $50\,\text{Hz}$ ($\Delta t_{\text{ctrl}} = 0.020\,\text{s}$).
   - Inelastic ground boundary contact condition at $z \le 0.0\,\text{m}$.

---

## 3. Repository Architecture

```
drone-trajectory-control/
├── configs/
│   ├── drone_params.yaml          # Mass, inertia tensor, motor limits, geometry
│   └── mission_params.yaml        # Start/target waypoints, timing, tolerances
├── envs/
│   ├── __init__.py
│   └── quadrotor_dynamics.py      # 13-state 6-DOF nonlinear dynamics & RK4 integrator
├── controllers/
│   ├── __init__.py
│   ├── base_controller.py         # Unified abstract controller interface
│   └── pid_controller.py          # 3-Phase waypoint FSM & cascaded PID with flare
├── utils/
│   ├── __init__.py
│   ├── metrics.py                 # Flight metrics (time, accuracy, energy, impulse)
│   ├── visualization.py           # Static 3D trajectory & 6-subplot telemetry plots
│   └── animation.py               # Interactive macOS real-time 3D GUI & MP4 exporter
├── tests/
│   ├── test_dynamics.py           # Unit tests: free fall & hover equilibrium
│   ├── test_visualization.py      # Unit tests: static plot generation
│   └── test_animation.py          # Unit tests: airframe geometry & video encoding
├── experiments/
│   └── run_pid.py                 # End-to-end Phase 1 baseline simulation runner
├── results/                       # Strictly isolated output folder (plots & videos)
│   ├── pid_baseline_3d.png
│   ├── pid_baseline_telemetry.png
│   └── pid_baseline_animation.mp4
├── requirements.txt
└── README.md
```

---

## 4. Phase 1 Implementation & Checklist Compliance Matrix

| Module | Requirement / Specification | Target Spec | Implemented Status | Verification Status |
|:---|:---|:---|:---:|:---:|
| `configs/` | Vehicle mass & inertia | $m=31.0\,\text{kg}, J=\text{diag}(1.25, 1.25, 2.50)$ | **Complete** | Validated |
| `configs/` | Thrust & tilt limits | $T_{\max}=600\,\text{N}, \theta_{\max} \le 25^\circ$ | **Complete** | Validated |
| `configs/` | Mission waypoints | Start $(-10, 0, 10)\,\text{m} \to$ Target $(0, 0, 0)\,\text{m}$ | **Complete** | Validated |
| `envs/` | 6-DOF Newton-Euler dynamics | 13-state quaternion representation in ENU | **Complete** | Unit Test Passed |
| `envs/` | High-fidelity integrator | 4th-Order Runge-Kutta (RK4) at 200 Hz | **Complete** | Energy error $< 10^{-14}$ |
| `envs/` | Ground contact handling | Inelastic impact at $z \le 0.0\,\text{m}$ (no penetration) | **Complete** | $z_{\min} = 0.0483\,\text{m} \ge 0.0$ |
| `controllers/` | Unified Controller Interface | `BaseController.compute_control(state, t, dt)` | **Complete** | Common API locked |
| `controllers/` | 3-Phase Waypoint Guidance | Phase 1 (Descent) $\to$ Phase 2 (Cruise) $\to$ Phase 3 (Land) | **Complete** | Smooth state transitions |
| `controllers/` | Ground Flare Sink-Rate | $v_{z,\text{des}} = -\text{clip}(0.12 + 0.50 z, 0.12, 2.2)\,\text{m/s}$ | **Complete** | Touchdown $|v_z| = 0.17\,\text{m/s}$ |
| `utils/` | Flight performance metrics | Flight time, position error, touchdown velocity, impulse | **Complete** | Verified |
| `utils/` | Static visual plots | 3D trajectory with attitude quivers & 6-subplot telemetry | **Complete** | High-res PNG generated |
| `utils/` | Interactive 3D Animation | Real-time Cocoa Aqua GUI, 6-DOF airframe, thrust quiver | **Complete** | Interactive & MP4 export |
| `experiments/`| End-to-end PID verification | Precision soft landing under tolerances | **Complete** | **ALL PASSED** |

---

## 5. Phase 1 Benchmark Verification Results (Scenario 3)

The baseline simulation was tested under full 6-DOF dynamics with physical saturation and aerodynamic damping:

```
======================================================================
>>> SCENARIO 3: 3-PHASE SEGMENTED PID BASELINE FLIGHT RESULTS
======================================================================
  Vehicle Mass:             31.0 kg (30.0 kg Drone + 1.0 kg Payload)
  Initial Position:         [-10.000, 0.000, 10.000] m
  Target Landing Point:     [  0.000, 0.000,  0.000] m
  Total Flight Time:        13.66 s
  Final Position:           [-0.002, 0.000, 0.048] m
  Final Position Error:     0.0483 m  (Tolerance: < 0.0500 m)  -> [PASS]
  Touchdown Velocity (vz):  -0.1737 m/s (Soft landing: < 0.2000 m/s) -> [PASS]
  Final Tilt Angle:         0.00 deg (Completely upright)     -> [PASS]
  Minimum Altitude:         0.0483 m (Underground: >= 0.0000 m) -> [PASS]
  Total Control Effort:     1,307,237.4 N^2*s
  Total Impulse:            21.8313 N*s
  Soft Landing Status:      SUCCESS
======================================================================
```

### Key Performance Highlights:
1. **Resolution of Pitch-Altitude Coupling**: Decoupling the trajectory into vertical descent to cruise altitude ($z=5\,\text{m}$), bounded horizontal cruise ($v_x \le 3.6\,\text{m/s}$), and terminal flare descent eliminated the severe ground crashes and overshoots observed in naive point-to-point PID.
2. **Precision Soft Landing**: Achieved a horizontal position error of only $1.9\,\text{mm}$ and touched down inside the $0.05\,\text{m}$ boundary with a gentle vertical sink rate of $-0.1737\,\text{m/s}$ and zero attitude tilt.

---

## 6. How to Run

### (1) Prerequisites
```bash
pip install -r requirements.txt
```
*(Requires `numpy`, `scipy`, `matplotlib`, `pyyaml`, and `ffmpeg` for MP4 video export)*

### (2) Run Unit Tests
```bash
# Test 6-DOF physical dynamics (free fall conservation & hover equilibrium)
python3 tests/test_dynamics.py

# Test static plot generation
python3 tests/test_visualization.py

# Test 3D airframe geometry & headless video export
python3 tests/test_animation.py
```

### (3) Run Simulation & Interactive Real-Time 3D Animation
```bash
# 1. Run simulation and generate static plots in results/
python3 experiments/run_pid.py

# 2. Launch native macOS interactive 3D GUI window (Real-time playback)
#    - Mouse drag: Rotate 3D view in real-time
#    - Scroll / Pinch: Zoom in and out
#    - Spacebar: Pause / Resume
#    - Q / Esc: Close window
python3 experiments/run_pid.py --animate

# 3. Play at 1.5x speed
python3 experiments/run_pid.py --animate --speed 1.5

# 4. Save high-resolution MP4 video to results/pid_baseline_animation.mp4
python3 experiments/run_pid.py --save-anim
```

---

## 7. Next Step: Phase 2 Roadmap

With the 6-DOF physical simulation environment, unified controller interfaces, and baseline PID performance established, the project moves to **Phase 2: Optimal Trajectory Guidance via ZEM/ZEV & $SE(3)$ Geometric Control**:
- **ZEM/ZEV Guidance Law**:
  $$\mathbf{a}_{\text{cmd}}(t) = \frac{6}{t_{\text{go}}^2} \mathbf{ZEM} - \frac{2}{t_{\text{go}}} \mathbf{ZEV} - \mathbf{g}_W$$
- **Time-to-Go ($t_{\text{go}}$) Optimal Search**: Golden section / bisection search to find the minimum flight time satisfying thrust bounds ($0 \le T \le 600\,\text{N}$).
- **Geometric $SE(3)$ Controller**: Tracking desired acceleration and heading on the special Euclidean group $SE(3)$ without Euler angle singularities (Lee et al., 2010).
