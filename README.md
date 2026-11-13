# Analytical & Numerical Simulator for H-Rotor VAWTs with Pitch Control

A scientific Python simulation framework developed to model, optimize, and analyze the stationary and transient aerodynamic behavior of H-Rotor Vertical Axis Wind Turbines (VAWT) using 2D Blade Element Theory (BET).

## 📌 Overview & Engineering Problem
Fixed-pitch VAWTs experience severe cyclic aerodynamic loads, structural fatigue, and self-starting challenges at low Tip Speed Ratios (\(\lambda\)). This framework derives and numerically solves the coupled dimensionless differential equations of rotor motion, accounting for:
* **Airfoil asymmetry & pitching moment (\(C_{m0}\))** using 360-degree extrapolated polars (`DU 06-W-200` & `NACA 0012`).
* **Pivot-point eccentricity (\(\tilde{d} = d/R\))** relative to the aerodynamic center.
* **Dynamic rotor moment of inertia \(\tilde{I}(\theta, \phi)\)** and mechanical/generator damping losses.

## 📊 Key Simulation Results

### 1. Stationary Parametric Analysis (\(C_P - \lambda\) & \(C_T - \lambda\))
Parametric sweep across fixed pitch angles \(\phi \in [0^\circ, 16^\circ]\) evaluating mean power and thrust performance:
![Stationary Parametric Curves](results/fig1_parametric_cp_ct.png)

### 2. Multi-Objective Azimuthal Pitch Optimization \(\phi(\theta)\)
Exhaustive vectorized optimization across attached angles of attack to maximize net aerodynamic torque (\(C_M\)) and minimize radial hub force magnitude (\(C_{Fa}\)):
![Power Maximization Pitch Law](results/fig2a_max_power_pitch.png)
![Hub Force Minimization Pitch Law](results/fig2b_minimized_cfa.png)

### 3. Transient Dynamics & Autorotation Convergence
Time-domain integration of the 3-blade coupled equations of motion, mapping convergence time \(\tilde{t}_{\mathrm{conv}}(\phi)\) and identifying the optimal fixed pitch angle (\(\phi = 4.60^\circ\)) for rapid steady-state autorotation:
![Transient Time Trajectories](results/fig3a_transient_trajectories.png)
![Autorotation Convergence & Stability](results/fig3b_autorotation_stability.png)

## 🗂️ Repository Structure
```text
├── data/                            # Public 360-degree airfoil polars (.dat)
├── src/
│   └── aerodynamics.py              # Polar interpolation & 2D BET kinematics
├── results/                         # High-resolution output figures (300 dpi)
├── 01_parametric_pitch.py           # Stationary Cp and Ct parametric sweeps
├── 02_pitch_optimization.py         # Azimuthal pitch laws phi(theta) optimization
├── 03_transient_autorotation.py     # Time-domain dynamics & stability mapping
└── requirements.txt                 # Python dependencies
```

## 🛠️ Tech Stack
`Python 3` · `NumPy` · `SciPy` · `Pandas` · `Matplotlib`
