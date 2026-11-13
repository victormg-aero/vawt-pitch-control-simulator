#    02_pitch_optimization.py 
"""
Azimuthal pitch law optimization phi(theta) for an H-Rotor VAWT.
Performs an exhaustive vectorized grid search across attached angles of attack
(-90 deg < alpha < 90 deg) for two primary engineering objectives:
  1. Aerodynamic torque / power maximization (max C_M) with pivot eccentricity & Cm.
  2. Radial hub force minimization (min CFa) to reduce cyclic structural fatigue.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.aerodynamics import (
    load_airfoil_polars,
    inflow_angle,
    dimensionless_relative_velocity,
    aerodynamic_force_coeffs,
)

# 1. Load 360-degree airfoil polars from QBlade .dat file
POLAR_FILE = "data/DU_06_W_200.dat"
cl_interp, cd_interp, cm_interp = load_airfoil_polars(POLAR_FILE)

# 2. Rotor geometric parameters (matching Chapter 4 baseline)
R = 5.0
c = 0.6
d = 0.25 * c  # Pivot offset (d = 0.15 m)

ds = d / R
cs = c / R

# 3. Calculation grids
lmb_grid = np.arange(0.0, 2.1, 0.4)                  # Tip Speed Ratio: 0.0 to 2.0
theta_grid = np.deg2rad(np.arange(0, 360, 1))        # Azimuthal angle: 0 to 359 deg
phi_range = np.deg2rad(np.linspace(-180, 180, 3601)) # Fine pitch grid (0.1 deg step)

n_theta = len(theta_grid)
n_lmb = len(lmb_grid)

# Result matrices
cm_max_mat = np.zeros((n_theta, n_lmb))
phi_max_power = np.zeros((n_theta, n_lmb))

cfa_min_mat = np.zeros((n_theta, n_lmb))
phi_min_cfa = np.zeros((n_theta, n_lmb))

# 4. Vectorized exhaustive search across lambda and theta
for k, lmb in enumerate(lmb_grid):
    for t, theta in enumerate(theta_grid):
        gam = inflow_angle(theta, lmb)
        alf = gam + phi_range

        # Filter physically attached angles of attack (-90 deg < alpha < 90 deg)
        valid_mask = (alf > np.deg2rad(-90)) & (alf < np.deg2rad(90))
        valid_phi = phi_range[valid_mask]
        valid_alf = alf[valid_mask]

        if len(valid_alf) == 0:
            continue

        # Aerodynamic hub forces (Cfx, Cfz) and resultant magnitude CFa
        _, _, cfx, cfz = aerodynamic_force_coeffs(theta, lmb, valid_phi, cl_interp, cd_interp)
        cfa = np.sqrt(cfz**2 + cfx**2)

        # Aerodynamic torque coefficient C_M (including eccentricity & pitching moment)
        cl_val = cl_interp(valid_alf)
        cd_val = cd_interp(valid_alf)
        cm_val = cm_interp(valid_alf)

        cp_arr = cl_val * np.cos(theta) - cd_val * (np.sin(theta) + lmb)
        urs = dimensionless_relative_velocity(theta, lmb)
        cm_arr = (
            urs * cp_arr
            + urs * ds * (
                cl_val * (np.sin(valid_alf) - lmb * np.cos(valid_phi))
                - cd_val * (np.cos(valid_alf) + lmb * np.sin(valid_phi))
            )
            - (urs**2) * cs * cm_val
        )

        # Objective 1: Maximize Aerodynamic Torque / Power
        idx_max_p = np.argmax(cm_arr)
        cm_max_mat[t, k] = cm_arr[idx_max_p]
        phi_max_power[t, k] = np.rad2deg(valid_phi[idx_max_p])

        # Objective 2: Minimize Radial Hub Force CFa
        idx_min_f = np.argmin(np.abs(cfa))
        cfa_min_mat[t, k] = cfa[idx_min_f]
        phi_min_cfa[t, k] = np.rad2deg(valid_phi[idx_min_f])

# 5. Publication-quality plotting
plt.style.use("seaborn-v0_8-whitegrid")
colors = plt.colormaps["plasma"](np.linspace(0, 0.9, n_lmb))
theta_deg = np.rad2deg(theta_grid)


def plot_pitch_and_metric(phi_data, metric_data, y_label_metric, title_left, title_right, save_path):
    """Generates clean, consistent dual-panel figures and exports at 300 dpi."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=110, constrained_layout=True)

    for i, lmb in enumerate(lmb_grid):
        ax0.plot(theta_deg, phi_data[:, i], linewidth=2.0, color=colors[i], label=f"{lmb:.1f}")
        ax1.plot(theta_deg, metric_data[:, i], linewidth=2.0, color=colors[i], label=f"{lmb:.1f}")

    ax0.set_xlim(0, 360)
    ax0.set_xlabel(r"Azimuthal Angle, $\theta$ [$^\circ$]", fontsize=12)
    ax0.set_ylabel(r"Optimal Pitch Angle, $\phi$ [$^\circ$]", fontsize=12)
    ax0.set_title(title_left, fontsize=13, fontweight="bold")
    ax0.legend(title=r"$\lambda$", loc="upper left", frameon=True, fontsize=10)
    ax0.grid(True, linestyle="--", alpha=0.6)

    ax1.set_xlim(0, 360)
    ax1.set_xlabel(r"Azimuthal Angle, $\theta$ [$^\circ$]", fontsize=12)
    ax1.set_ylabel(y_label_metric, fontsize=12)
    ax1.set_title(title_right, fontsize=13, fontweight="bold")
    ax1.legend(title=r"$\lambda$", loc="upper left", frameon=True, fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)

    fig.savefig(save_path, dpi=300, bbox_inches="tight")


# Figure 2A: Maximization of Power / Torque
plot_pitch_and_metric(
    phi_max_power,
    cm_max_mat,
    r"Aerodynamic Torque Coeff., $C_M$ [-]",
    r"Pitch Law for Power Maximization ($\max C_M$)",
    r"Maximized Aerodynamic Torque $C_M(\theta)$",
    "results/fig2a_max_power_pitch.png",
)

# Figure 2B: Minimization of CFa
plot_pitch_and_metric(
    phi_min_cfa,
    cfa_min_mat,
    r"Hub Radial Force Coeff., $C_{Fa}$ [-]",
    r"Pitch Law for $C_{Fa}$ Minimization",
    r"Minimized Hub Force $C_{Fa}(\theta)$",
    "results/fig2b_minimized_cfa.png",
)

plt.show()
