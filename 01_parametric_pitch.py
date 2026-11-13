"""
Parametric aerodynamic analysis of an H-Rotor VAWT under fixed pitch angles (phi).
Generates publication-quality Cp-TSR and Ct-TSR performance curves.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.aerodynamics import load_airfoil_polars, aerodynamic_force_coeffs

# 1. Load 360-degree airfoil polars from QBlade .dat file
POLAR_FILE = "data/DU_06_W_200.dat"
cl_interp, cd_interp, _ = load_airfoil_polars(POLAR_FILE)

# 2. Simulation grids (TSR capped at 2.8 within valid non-induced BET regime)
tsr_grid = np.arange(0.0, 2.85, 0.05)
theta_grid = np.deg2rad(np.arange(0, 360, 1))
phi_grid = np.deg2rad(np.arange(0, 17, 2))  # Fixed pitch angles: 0 to 16 deg

n_tsr = len(tsr_grid)
n_phi = len(phi_grid)

cp_mean = np.zeros((n_tsr, n_phi))
ct_mean = np.zeros((n_tsr, n_phi))

# 3. Parametric sweep over pitch angle (phi) and Tip Speed Ratio (lambda)
for j, phi in enumerate(phi_grid):
    for k, tsr in enumerate(tsr_grid):
        cn, ct, _, _ = aerodynamic_force_coeffs(theta_grid, tsr, phi, cl_interp, cd_interp)

        # Instantaneous thrust and power coefficients
        c_thrust = cn * np.sin(theta_grid) - ct * np.cos(theta_grid)
        alpha = np.arctan2(np.cos(theta_grid), np.sin(theta_grid) + tsr) + phi
        c_power = cl_interp(alpha) * np.cos(theta_grid) - cd_interp(alpha) * (np.sin(theta_grid) + tsr)

        ct_mean[k, j] = np.mean(c_thrust)
        cp_mean[k, j] = np.mean(c_power)

# 4. Publication-quality plotting (sized to fit cleanly on screen)
plt.style.use("seaborn-v0_8-whitegrid")
colors = plt.colormaps["viridis"](np.linspace(0, 0.9, n_phi))

fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=110, constrained_layout=True)

# Subplot 1: Power Coefficient (Cp) vs TSR
for j in range(n_phi):
    axes[0].plot(
        tsr_grid,
        cp_mean[:, j],
        linewidth=2.0,
        color=colors[j],
        label=rf"{np.rad2deg(phi_grid[j]):.0f}$^\circ$"
    )

axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[0].set_xlim(0, 2.8)
axes[0].set_xlabel(r"Tip Speed Ratio, $\lambda$ [-]", fontsize=12)
axes[0].set_ylabel(r"Mean Power Coefficient, $\bar{C}_P$ [-]", fontsize=12)
axes[0].set_title(r"Power Coefficient vs. $\lambda$", fontsize=13, fontweight="bold")
axes[0].legend(title=r"Pitch $\phi$", frameon=True, ncol=2, fontsize=9)
axes[0].grid(True, linestyle="--", alpha=0.6)

# Subplot 2: Thrust Coefficient (Ct) vs TSR
for j in range(n_phi):
    axes[1].plot(
        tsr_grid,
        ct_mean[:, j],
        linewidth=2.0,
        color=colors[j],
        label=rf"{np.rad2deg(phi_grid[j]):.0f}$^\circ$"
    )

axes[1].set_xlim(0, 2.8)
axes[1].set_xlabel(r"Tip Speed Ratio, $\lambda$ [-]", fontsize=12)
axes[1].set_ylabel(r"Mean Thrust Coefficient, $\bar{C}_T$ [-]", fontsize=12)
axes[1].set_title(r"Thrust Coefficient vs. $\lambda$", fontsize=13, fontweight="bold")
axes[1].legend(title=r"Pitch $\phi$", frameon=True, ncol=2, fontsize=9)
axes[1].grid(True, linestyle="--", alpha=0.6)

plt.savefig("results/fig1_parametric_cp_ct.png", dpi=300, bbox_inches="tight")
plt.show()
