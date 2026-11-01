# 6-DOF Autonomous Drone Trajectory Control & Precision Landing Benchmark

> **KAIST Flight Dynamics & Guidance, Navigation, and Control (GNC) Research Project**  
> **Author**: Sungcheol Ha (KAIST Department of Aerospace Engineering & Physics)  
> **Reference Origin**: Modernized & upgraded from legacy Point-Mass MATLAB codebase (`sungchol3/RL-base-drone-delivery`) to a high-fidelity 6-DOF Python simulation platform.

---

## 1. Project Overview

This research repository implements a benchmark framework for **optimal trajectory tracking and autonomous precision landing of a high-mass (31.0 kg) multirotor UAV**. The project compares classical and modern intelligent flight guidance and control paradigms across identical high-fidelity 6-DOF physical simulation environments:

1. **Phase 1 (Baseline)**: Cascaded 3-Phase Decoupled PID Controller with Ground Flare Regulation. `[COMPLETED]`
2. **Phase 2 (Optimal Guidance)**: Zero-Effort-Miss / Zero-Effort-Velocity (ZEM/ZEV) Guidance with $SE(3)$ Geometric Tracking Control. `[COMPLETED]`
3. **Phase 3 (Optimal Control)**: Nonlinear Model Predictive Control (NMPC) with Rotor Input and State Constraints. `[COMPLETED]`
4. **Phase 4 (Deep RL)**: Mission-Tailored Reinforcement Learning (Soft Actor-Critic, SAC & PPO). `[NEXT]`
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

   $$m \dot{\mathbf{v}}_W = m \mathbf{g}_W + \mathbf{R}_{WB}(\mathbf{q}) \begin{bmatrix} 0 \cr 0 \cr T \end{bmatrix} - \mathbf{D}_v \mathbf{v}_W$$

   where $m = 31.0\,\text{kg}$ ($30.0\,\text{kg}$ airframe + $1.0\,\text{kg}$ delivery payload), $T$ is total rotor thrust, and $\mathbf{D}_v = \text{diag}(0.1, 0.1, 0.1)\,\text{N}\cdot\text{s/m}$ is aerodynamic damping.

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
├── .gitignore                     # Git tracking exclusions (virtualenvs, cache, media)
├── README.md                      # System architecture, benchmarks, and run guide
├── requirements.txt               # Python package dependencies
├── configs/
│   ├── drone_params.yaml          # Mass, inertia tensor, motor limits, geometry
│   └── mission_params.yaml        # Start/target waypoints, timing, tolerances
├── envs/
│   ├── __init__.py
│   └── quadrotor_dynamics.py      # 13-state 6-DOF nonlinear dynamics & 200 Hz RK4
├── controllers/
│   ├── __init__.py
│   ├── base_controller.py         # Unified abstract controller interface
│   ├── pid_controller.py          # Phase 1: 3-Phase waypoint FSM & cascaded PID with flare
│   ├── zem_zev_guidance.py        # Phase 2: ZEM/ZEV guidance & bisection Tf* solver
│   ├── geometric_controller.py    # Phase 2: SE(3) geometric attitude/thrust tracking
│   ├── zem_zev_controller.py      # Phase 2: Unified ZEM/ZEV trajectory controller
│   ├── nmpc_casadi_model.py       # Phase 3: CasADi Multiple Shooting OCP builder
│   └── nmpc_controller.py         # Phase 3: Receding horizon NMPC & SE(3) controller
├── experiments/
│   ├── run_pid.py                 # Phase 1 baseline simulation & benchmark runner
│   ├── run_zem_zev.py             # Phase 2 ZEM/ZEV simulation & comparative benchmark
│   └── run_nmpc.py                # Phase 3 CasADi NMPC simulation & 3-way benchmark
├── tests/
│   ├── test_dynamics.py           # Dynamics unit tests: free fall & hover equilibrium
│   ├── test_visualization.py      # Plot generation pipeline unit tests
│   ├── test_animation.py          # 3D airframe geometry & video encoding unit tests
│   ├── test_zem_zev.py            # ZEM/ZEV analytical equivalence & SE(3) stability tests
│   └── test_nmpc.py               # NMPC OCP feasibility, hard constraints & timing tests
├── utils/
│   ├── __init__.py
│   ├── metrics.py                 # Flight metrics (time, accuracy, energy, impulse)
│   ├── visualization.py           # Static 3D trajectory & 6-subplot telemetry plots
│   └── animation.py               # Interactive macOS real-time 3D GUI & MP4 exporter
└── results/
    └── .gitkeep                   # Output folder for generated plots & MP4 animation videos
```

---

## 4. Phase 1 Implementation: Baseline Decoupled PID

### (1) Problem Formulation & Key Findings
In legacy MATLAB simulations, 3-DOF point-mass approximations relied on artificial velocity clamping (`vel(1:2) = 0` and `pos(3) = z_hold`). When deployed on high-mass ($31.0\,\text{kg}$) 6-DOF dynamics, naive point-to-point PID suffered from severe **pitch-altitude coupling**: pitching down to accelerate accumulated excessive horizontal momentum ($p_x \approx 280\,\text{kg}\cdot\text{m/s}$), and braking reverse-pitching caused immediate loss of vertical lift ($T\cos\theta$), plunging the drone into ground crashes ($z \le 0$).

### (2) 3-Phase Decoupling & Ground Flare Solution
1. **Phase 1 (Vertical Descent)**: $(-10, 0, 10) \to (-10, 0, 5)\,\text{m}$ without pitching.
2. **Phase 2 (Horizontal Cruise)**: Altitude locked at $z = 5.0\,\text{m}$ with cruise speed bounded to $v_x \le 3.6\,\text{m/s}$. Horizontal velocity fully arrested over the target pad ($|v_x| < 0.6\,\text{m/s}$).
3. **Phase 3 (Precision Flare Landing)**: Pure vertical descent over pad with altitude-dependent flare sink-rate regulation:

   $$v_{z,\text{des}}(z) = -\text{clip}(0.12 + 0.50 z, \; 0.12, \; 2.2)\quad [\text{m/s}]$$

### (3) Baseline Verification Scoreboard
```
======================================================================
>>> SCENARIO 3: 3-PHASE SEGMENTED PID BASELINE FLIGHT RESULTS
======================================================================
  Vehicle Mass:             31.0 kg (30.0 kg Drone + 1.0 kg Payload)
  Initial Station:          [-10.000, 0.000, 10.000] m
  Target Landing Point:     [  0.000, 0.000,  0.000] m
  Total Flight Time:        13.66 s
  Final Landing Position:   [-0.002, 0.000, 0.048] m
  Final Position Error:     0.0483 m  (Tolerance: < 0.0500 m)  -> [PASS]
  Touchdown Velocity (vz):  -0.1737 m/s (Soft landing: < 0.2000 m/s) -> [PASS]
  Final Tilt Angle:         0.00 deg (Completely upright)     -> [PASS]
  Minimum Altitude:         0.0483 m (Underground: >= 0.0000 m) -> [PASS]
  Total Control Effort:     1,307,237.4 N^2*s
  Total Impulse:            21.8313 N*s
  Soft Landing Status:      SUCCESS
======================================================================
```

---

## 5. Phase 2 Implementation: ZEM/ZEV Optimal Guidance & $SE(3)$ Geometric Control

### (1) Mathematical Formulation & Core Algorithms
Phase 2 upgrades trajectory generation from heuristic waypoint FSMs to analytical energy-optimal powered descent guidance:

1. **Closed-Form ZEM/ZEV Guidance**:

   $$\mathbf{a}_\text{thrust}(t) = \frac{6}{t_\text{go}^2} \mathbf{ZEM}(t) - \frac{2}{t_\text{go}} \mathbf{ZEV}(t)$$

   where:

   $$\mathbf{ZEM}(t) = \mathbf{p}_f - \left( \mathbf{p}(t) + \mathbf{v}(t) t_{\text{go}} + \frac{1}{2} \mathbf{g}_W t_{\text{go}}^2 \right)$$

   $$\mathbf{ZEV}(t) = \mathbf{v}_f - \left( \mathbf{v}(t) + \mathbf{g}_W t_{\text{go}} \right)$$

2. **$O(1)$ Convex Peak Acceleration & Bisection $T_f^*$ Solver**:

   Because $\mathbf{a}_{\text{thrust}}(\tau) = \mathbf{c}_0 + \mathbf{c}_1 \tau$ is affine in time, its norm squared is strictly convex. The maximum acceleration on $\tau \in [0, t_{\text{go}}]$ occurs strictly at the boundaries:

   $$\max_{\tau \in [0, t_{\text{go}}]} \|\mathbf{a}_{\text{thrust}}(\tau)\| = \max\left( \|\mathbf{a}_{\text{thrust}}(0)\|, \; \|\mathbf{a}_{\text{thrust}}(t_{\text{go}})\| \right)$$

   where:

   $$\mathbf{a}_{\text{thrust}}(0) = \frac{6}{t_{\text{go}}^2}\mathbf{ZEM} - \frac{2}{t_{\text{go}}}\mathbf{ZEV}, \quad \mathbf{a}_{\text{thrust}}(t_{\text{go}}) = -\frac{6}{t_{\text{go}}^2}\mathbf{ZEM} + \frac{4}{t_{\text{go}}}\mathbf{ZEV}$$

   This enables instantaneous $O(1)$ feasibility checks, allowing a bisection search to determine the minimum flight time $T_f^*$ subject to $a_{\max} \approx 15.85\,\text{m/s}^2$ ($T_{\max} = 600\,\text{N}$, $\theta_{\max} = 35^\circ$) in $< 15$ iterations.

3. **$SE(3)$ Geometric Tracking Controller (Lee et al., 2010)**:
   - Orthonormal basis on $SO(3)$: $\mathbf{b}_{3,d} = \mathbf{a}_{\text{thrust}} / \|\mathbf{a}_{\text{thrust}}\|$, $\mathbf{b}_{2,d} = (\mathbf{b}_{3,d} \times \mathbf{b}_{1,c}) / \|\mathbf{b}_{3,d} \times \mathbf{b}_{1,c}\|$, $\mathbf{b}_{1,d} = \mathbf{b}_{2,d} \times \mathbf{b}_{3,d}$.
   - Collective thrust: $T = m (\mathbf{a}_{\text{thrust}} \cdot \mathbf{b}_3) \in [0, 600]\,\text{N}$.
   - Attitude tracking error on Lie Group $SO(3)$: $\mathbf{e}_R = \frac{1}{2} (\mathbf{R}_d^T \mathbf{R} - \mathbf{R}^T \mathbf{R}_d)^\vee$.
   - Control torque with gyroscopic compensation: $\boldsymbol{\tau} = -k_R \mathbf{e}_R - k_\Omega \mathbf{e}_\Omega + \boldsymbol{\Omega} \times (\mathbf{J}\boldsymbol{\Omega})$.

### (2) Phase 2 Unit Test Verification (`tests/test_zem_zev.py`)
```
============================================================
>>> Running Phase 2 ZEM/ZEV & SE(3) Unit Tests
============================================================
[PASS] Scenario 1-A (Analytical Machine Precision): 
       - Terminal position error: 3.17e-14 m (Exact closed-form equivalence)
       - Terminal velocity error: 7.41e-15 m/s
[PASS] Scenario 1-B (Optimal Tf Bisection Solver): 
       - Solved Tf* = 3.365 s (Peak accel: 15.84 <= 15.85 m/s^2 bound)
[PASS] Scenario 2 (SE(3) Attitude Tracking & Lyapunov Stability): 
       - Initial large tilt (Roll 20°, Pitch 30°) converges in < 1.0 s
       - Error metric: Psi = 2.28e-06, ||omega|| = 7.28e-03 rad/s
[PASS] Scenario 3 (Singularity Avoidance & Boundedness): 
       - Guidance remains strictly bounded as t_go -> 0 without NaN/Inf
============================================================
ALL PHASE 2 UNIT TESTS PASSED SUCCESSFULLY!
============================================================
```

---

## 6. Phase 3 Implementation: CasADi NMPC with Hard Constraints

### (1) Mathematical Formulation & OCP Structure
Phase 3 formulates finite-horizon optimal control directly handling state and actuator boundaries:
1. **Multiple Shooting Discretization**:
   - 4th-order Runge-Kutta (RK4) integration defect constraints:

   $$\mathbf{x}_{k+1} - \mathbf{F}_{\text{RK4}}(\mathbf{x}_k, \mathbf{u}_k, \Delta t) = \mathbf{0}, \quad k = 0, \dots, N-1$$

2. **Hard Inequality Constraints**:
   - Ground collision avoidance: $z_k \ge z_{\min} = 0.04\,\text{m}$ ($\forall k$)
   - 2nd-order cone tilt angle limit: $\sqrt{u_{x,k}^2 + u_{y,k}^2} \le u_{z,k} \tan(35^\circ)$
   - Total motor thrust saturation: $\|\mathbf{u}_k\| \le \frac{T_{\max}}{m} \approx 19.35\,\text{m/s}^2$
   - Terminal soft touchdown velocity: $v_{z,N} \ge -0.20\,\text{m/s}$
3. **Warm-Starting & Glideslope Reference Governor**:
   - Solution shifting ( $\mathbf{w}_{\text{init}} = [\mathbf{x}^*_{1:N}, \mathbf{u}^*_{1:N-1}]$ ) enables fast $< 16\,\text{ms}$ mean IPOPT solve times ($64.7\,\text{Hz}$ RHC).
   - Altitude reference governor maintains safe glideslope buffer until pad alignment ($r_{xy} \le 0.25\,\text{m}$) before initiating terminal flare soft touchdown.

### (2) Phase 3 Unit Test Verification (`tests/test_nmpc.py`)
```
============================================================
>>> Running Phase 3 NMPC Unit Tests & Visualization Checks
============================================================
[PASS] Scenario 1: OCP cold start converged (58.41 ms, status: Solve_Succeeded)
[PASS] Scenario 2: All hard constraints strictly satisfied:
       - Minimum altitude across horizon: 7.2559 m >= 0.0400 m
       - Peak control acceleration: 19.3548 m/s^2 <= 19.3548 m/s^2
       - Peak predicted tilt angle: 35.00 deg <= 35.00 deg
[PASS] Scenario 3: Warm-start real-time efficiency verified (Mean: 12.77 ms, Rate: 78.3 Hz)
[PASS] Scenario 4: Terminal soft touchdown constraint satisfied (vz,N = -0.2000 m/s >= -0.2000 m/s)
[PASS] Scenario 5: NMPCController BaseController interface & statistics verified
[PASS] Scenario 6: 3D Trajectory visual generated and verified at results/test_nmpc_3d.png
============================================================
ALL PHASE 3 NMPC UNIT TESTS PASSED SUCCESSFULLY!
============================================================
```

---

## 7. Comprehensive 3-Way Comparative Benchmark: PID vs ZEM/ZEV vs NMPC

### (1) NMPC Optimization & Reliability Diagnostics
In the 6-DOF closed-loop simulation, the NMPC solver demonstrated flawless convergence reliability:
```
------------------------------------------------------------------------------------------
           NMPC OPTIMIZATION CONVERGENCE & FALLBACK RELIABILITY STATISTICS
------------------------------------------------------------------------------------------
Total NMPC Solver Invocations:         84 calls
Optimal Solutions Succeeded:           84 / 84 (100.0%) -> [PASS]
Fallback to ZEM/ZEV Guidance:           0 / 84 (  0.0%) -> [PASS]
Terminal Precision Flare Steps:        84 steps (z <= 0.80 m)
Average Solve Time (Warm-Start):     15.46 ms (64.7 Hz effective rate)
Peak Solve Time across Mission:      32.76 ms (< 35.0 ms real-time ceiling)
------------------------------------------------------------------------------------------
```

### (2) Quantitative Benchmark Scoreboard
The table below presents the quantitative benchmark results across all three control paradigms under identical 6-DOF simulation conditions (`start_pos = [-10, 0, 10] m`, `target_pos = [0, 0, 0] m`, mass $31.0\,\text{kg}$, 200 Hz RK4 integration):

| Performance Metric | Phase 1 (Baseline PID) | Phase 2 (ZEM/ZEV + $SE(3)$) | Phase 3 (CasADi NMPC) | Improvement vs PID | Evaluation Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Total Flight Time ($T_f$)** | $13.66\,\text{s}$ | $4.94\,\text{s}$ | **$5.02\,\text{s}$** | **$+63.3\%$ (FASTER)** | **PASS** ($< 6.0\,\text{s}$) |
| **Landing Position Error** | $0.0483\,\text{m}$ | $0.0487\,\text{m}$ | **$0.0497\,\text{m}$** | **$4.97\,\text{cm}$ (Radial error $1.0\,\text{cm}$)** | **PASS** ($< 0.05\,\text{m}$) |
| **Touchdown Velocity ($v_z$)** | $-0.1737\,\text{m/s}$ | $-0.1649\,\text{m/s}$ | **$-0.1555\,\text{m/s}$** | Ideal soft landing ($< 0.20\,\text{m/s}$) | **PASS** |
| **Final Attitude Tilt Angle** | $0.00^\circ$ | $0.34^\circ$ | **$0.20^\circ$** | Upright terminal landing | **PASS** ($< 2.0^\circ$) |
| **Minimum Altitude ($z_{\min}$)** | $0.0483\,\text{m}$ | $0.0481\,\text{m}$ | **$0.0487\,\text{m}$** | Zero ground crash ($\ge 0.0\,\text{m}$) | **PASS** |
| **Total Control Effort ($\int T^2 dt$)** | $1,307,237.4\,\text{N}^2\cdot\text{s}$ | $495,969.8\,\text{N}^2\cdot\text{s}$ | **$621,136.1\,\text{N}^2\cdot\text{s}$** | **$+52.5\%$ (ENERGY SAVED)** | **PASS** |
| **Total Input Impulse** | $21.8313\,\text{N}\cdot\text{s}$ | $14.7514\,\text{N}\cdot\text{s}$ | **$52.9684\,\text{N}\cdot\text{s}$** | Active receding horizon control | **PASS** |
| **NMPC Optimal Success Rate** | N/A | N/A | **$100.0\%$ (84/84 Solved)** | Zero IPOPT divergence | **PASS** |
| **ZEM/ZEV Fallback Rate** | N/A | N/A | **$0.0\%$ (0/84 Switched)** | 100% pure NMPC flight | **PASS** |
| **Mean Solve Time (Warm)** | N/A (Analytical) | $< 0.1\,\text{ms}$ ($O(1)$) | **$15.46\,\text{ms}$** | Real-time feasible ($64.7\,\text{Hz}$) | **PASS** ($< 25\,\text{ms}$) |
| **Peak Solve Time** | N/A | $< 0.1\,\text{ms}$ | **$32.76\,\text{ms}$** | Strict upper bound | **PASS** ($< 35\,\text{ms}$) |
| **Hard Constraint Enforcement** | Heuristic clipping | Heuristic saturation | **Exact Optimization Constraints** | Mathematically guaranteed | **SUPERIOR** |
| **Mission Success Status** | **SUCCESS** | **SUCCESS** | **SUCCESS** | **All Benchmark Criteria Met** | **ALL PASSED** |

---

## 8. How to Run

### (1) Setup Environment
```bash
# Install required Python packages
pip install -r requirements.txt
```

### (2) Execute Unit Tests
```bash
# Phase 1: Physical dynamics & energy conservation tests
python3 tests/test_dynamics.py

# Phase 1: Visualization and animation integrity tests
python3 tests/test_visualization.py
python3 tests/test_animation.py

# Phase 2: ZEM/ZEV analytical equivalence & SE(3) Lyapunov stability tests
python3 tests/test_zem_zev.py

# Phase 3: CasADi NMPC OCP feasibility, hard constraints & timing tests
python3 tests/test_nmpc.py
```

### (3) Run Simulations & Interactive 3D Animations

#### 1. Phase 1: Baseline 3-Phase PID Simulation
```bash
# Run simulation and save plots to results/
python3 experiments/run_pid.py

# Launch native macOS interactive real-time 3D GUI window
python3 experiments/run_pid.py --animate

# Save high-resolution MP4 video to results/pid_baseline_animation.mp4
python3 experiments/run_pid.py --save-anim
```

#### 2. Phase 2: ZEM/ZEV Optimal Guidance Simulation
```bash
# Run simulation and save plots to results/
python3 experiments/run_zem_zev.py

# Launch native macOS interactive real-time 3D GUI window
python3 experiments/run_zem_zev.py --animate

# Play at 1.5x speed
python3 experiments/run_zem_zev.py --animate --speed 1.5

# Save high-resolution MP4 video to results/zem_zev_animation.mp4
python3 experiments/run_zem_zev.py --save-anim
```

#### 3. Phase 3: CasADi Nonlinear Model Predictive Control (NMPC)
```bash
# Run closed-loop NMPC simulation (automatically saves plots and MP4 video)
python3 experiments/run_nmpc.py

# Launch native macOS interactive real-time 3D GUI window
python3 experiments/run_nmpc.py --animate

# Directly view generated high-resolution MP4 animation video on macOS
open results/nmpc_animation.mp4
```

---

## 9. Next Step: Phase 4 Roadmap (Deep Reinforcement Learning & Adaptive Control)

Building on the analytical and numerical optimal control foundations from Phases 1–3, **Phase 4 introduces Deep Reinforcement Learning (DRL)**:
- **Algorithms**: Soft Actor-Critic (SAC) & Proximal Policy Optimization (PPO).
- **Environment**: Gymnasium-compliant 6-DOF multirotor environment wrapping `Quadrotor6DOF`.
- **Hierarchical Separation**: RL agent outputs high-level acceleration commands or reference attitudes, while inner $SE(3)$ tracking controller handles 200 Hz attitude stabilization.
- **Domain Randomization**: Mass variations ($\pm 20\%$), wind gusts (von Kármán turbulence), and actuator delay to ensure sim-to-real transferability.
