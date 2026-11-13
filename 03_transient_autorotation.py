# 03_transient_autorotation.py 
"""
Transient dynamic simulation and autorotation convergence analysis for a 3-blade H-Rotor VAWT.
Faithful implementation of Chapter 5 dimensionless equations of motion.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.aerodynamics import load_airfoil_polars

# === 1. Load 360-degree airfoil polars ===
POLAR_FILE = "data/NACA_0012_Re1.000_360.dat"
Cl_interp, Cd_interp, Cm_interp = load_airfoil_polars(POLAR_FILE)

# === 2. Physical and dimensionless parameters (exact TFG values) ===
R = 5
U = 10
N = 3
rho = 1.225

d = 0.5
c = 0.6
m = 100
g = 9.81
W = m * g

mur = 0.1
mu2 = 0.15
mut = 0.2

ds = d / R
cs = c / R
ms = m / (rho * (R**3))
gs = g * R / (U**2)
Ws = ms * gs

Rs = 1
Us = 1


# === 3. Exact TFG aerodynamic and inertia functions (with phi properly in radians) ===
def urs(theta, lmb):
    return np.sqrt(lmb**2 + 2 * lmb * np.sin(theta) + 1)


def gamma(theta, lmb):
    return np.arctan2(np.cos(theta), np.sin(theta) + lmb)


def CM(phi_deg, theta, lmb, cs, ds, mur, mut, mu2, N, Ws):
    phi_rad = np.deg2rad(phi_deg)
    gam = gamma(theta, lmb)
    alf = gam + phi_rad

    if alf <= -np.pi or alf >= np.pi:
        return 0.0

    cl_val = Cl_interp(alf)
    cd_val = Cd_interp(alf)
    cm_val = Cm_interp(alf)

    Cn = cd_val * np.sin(gam) + cl_val * np.cos(gam)
    Ct = cl_val * np.sin(gam) - cd_val * np.cos(gam)

    C_P = cl_val * np.cos(theta) - cd_val * (np.sin(theta) + lmb)
    Fz = Cn * np.sin(theta) + Ct * np.cos(theta)
    Fx = Cn * np.cos(theta) - Ct * np.sin(theta)
    Fa = np.sqrt(Fz**2 + Fx**2)

    Urs = urs(theta, lmb)

    C_M = (
        Urs * C_P
        + Urs * ds * (
            cl_val * (np.sin(alf) - lmb * np.cos(phi_rad))
            - cd_val * (np.cos(alf) + lmb * np.sin(phi_rad))
        )
        - Urs**2 * cs * cm_val
    )

    return C_M - mur * Ws / N - mu2 * Fa / N - (C_M - mur * Ws / N - mu2 * Fa / N) * mut / N


def MI(ms, cs, Rs, theta, phi_deg, CM_val):
    phi_rad = np.deg2rad(phi_deg)
    Ios = ms * cs**2 / 12 + ms * (
        (Rs * np.cos(theta) - ds * np.sin(theta + phi_rad)) ** 2
        + (Rs * np.sin(theta) + ds * np.cos(theta - phi_rad)) ** 2
    )
    return CM_val / Ios


# === 4. Time integration ===
phi = np.arange(0, 7.1, 0.1)
l0 = len(phi)

T = 3 * 60
Ts = T * R / U
dt = 0.1
dts = dt * R / U
dtsl = np.arange(0, Ts, dts)
l1 = len(dtsl)

theta = np.zeros((l1, l0))
lmb = np.zeros((l1, l0))
lmb[0, :] = 1  # Initial impulse

print("Running transient simulation...")
for k in range(l0):
    for j in range(1, l1):
        CM1 = CM(phi[k], theta[j - 1, k], lmb[j - 1, k], cs, ds, mur, mut, mu2, N, Ws)
        CM2 = CM(phi[k], theta[j - 1, k] + np.deg2rad(120), lmb[j - 1, k], cs, ds, mur, mut, mu2, N, Ws)
        CM3 = CM(phi[k], theta[j - 1, k] + np.deg2rad(240), lmb[j - 1, k], cs, ds, mur, mut, mu2, N, Ws)

        MI1 = MI(ms, cs, Rs, theta[j - 1, k], phi[k], CM1)
        MI2 = MI(ms, cs, Rs, theta[j - 1, k] + np.deg2rad(120), phi[k], CM2)
        MI3 = MI(ms, cs, Rs, theta[j - 1, k] + np.deg2rad(240), phi[k], CM3)

        MIT = MI1 + MI2 + MI3

        theta[j, k] = theta[j - 1, k] + dts * lmb[j - 1, k]
        lmb[j, k] = lmb[j - 1, k] + dts * MIT


# === 5. Figure 1: Time evolution of theta and lambda ===
plt.style.use("seaborn-v0_8-whitegrid")
fig, (ax_th, ax_lm) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=110, constrained_layout=True)

for k in range(l0):
    if np.isclose(phi[k], 1.0, atol=1e-6) or np.isclose(phi[k], 3.0, atol=1e-6) or np.isclose(phi[k], 5.0, atol=1e-6):
        ax_th.plot(dtsl, theta[:, k], linewidth=2.0, label=rf"$\phi = {phi[k]:.1f}^\circ$")
        ax_lm.plot(dtsl, lmb[:, k], linewidth=2.0, label=rf"$\phi = {phi[k]:.1f}^\circ$")

ax_th.set_xlabel(r"Dimensionless Time, $\tilde{t}$ [-]", fontsize=12)
ax_th.set_ylabel(r"Azimuthal Angle, $\theta$ [rad]", fontsize=12)
ax_th.set_title(r"Time Evolution of $\theta(\tilde{t})$", fontsize=13, fontweight="bold")
ax_th.grid(True, linestyle="--", alpha=0.6)
ax_th.legend()

ax_lm.set_xlabel(r"Dimensionless Time, $\tilde{t}$ [-]", fontsize=12)
ax_lm.set_ylabel(r"Tip Speed Ratio, $\lambda$ [-]", fontsize=12)
ax_lm.set_title(r"Time Evolution of $\lambda(\tilde{t})$", fontsize=13, fontweight="bold")
ax_lm.grid(True, linestyle="--", alpha=0.6)
ax_lm.legend()

fig.savefig("results/fig3a_transient_trajectories.png", dpi=300, bbox_inches="tight")


# === 6. Convergence detection ===
epsilon = 1e-2
consec = 50
lambda_min = 0.01

t_convergencia = []
lambda_final = []

for k in range(lmb.shape[1]):
    lmb_k = lmb[:, k]
    converged_at = None

    for j in range(consec, len(lmb_k)):
        prev_values = lmb_k[j - consec : j]
        delta = np.abs(prev_values - lmb_k[j])
        if np.all(delta < epsilon):
            converged_at = dtsl[j]
            break

    if converged_at is None:
        converged_at = np.inf

    lambda_final_val = lmb_k[-1]
    lambda_final.append(lambda_final_val)

    if np.abs(lambda_final_val) > lambda_min:
        t_convergencia.append(converged_at)
    else:
        t_convergencia.append(np.inf)

t_convergencia = np.array(t_convergencia)
lambda_final = np.array(lambda_final)

idx_opt = np.argmin(t_convergencia)
phi_opt = phi[idx_opt]


# === 7. Figure 2: Convergence Time & Final Lambda vs. Pitch ===
fig2, (ax_conv, ax_fin) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=110, constrained_layout=True)

phi_validos = phi[np.isfinite(t_convergencia)]
t_validos = t_convergencia[np.isfinite(t_convergencia)]

ax_conv.scatter(phi_validos, t_validos, color="navy", label="Detected Convergence", zorder=3)
if np.isfinite(t_convergencia[idx_opt]):
    ax_conv.axvline(phi_opt, linestyle="--", color="green", linewidth=2.0, label=rf"Optimal $\phi = {phi_opt:.2f}^\circ$", zorder=2)

ax_conv.set_xlabel(r"Fixed Pitch Angle, $\phi$ [$^\circ$]", fontsize=12)
ax_conv.set_ylabel(r"Convergence Time, $\tilde{t}_{\mathrm{conv}}$ [-]", fontsize=12)
ax_conv.set_title(r"Convergence Time of $\lambda$ vs. $\phi$", fontsize=13, fontweight="bold")
ax_conv.grid(True, linestyle="--", alpha=0.6)
ax_conv.legend()

ax_fin.plot(phi, lambda_final, marker="s", markersize=4, linewidth=2.0, color="darkorange", label=r"Final $\lambda$")
ax_fin.axhline(lambda_min, linestyle="--", color="red", label=rf"Lower Threshold = {lambda_min}")

valid_mask = lambda_final > lambda_min
if np.any(valid_mask):
    phi_valid = phi[valid_mask]
    ax_fin.axvline(np.min(phi_valid), color="green", linestyle="--", linewidth=2.0, label="Valid Region Start")
    ax_fin.axvline(np.max(phi_valid), color="green", linestyle="--", linewidth=2.0, label="Valid Region End")

ax_fin.set_xlabel(r"Fixed Pitch Angle, $\phi$ [$^\circ$]", fontsize=12)
ax_fin.set_ylabel(r"Final Tip Speed Ratio, $\lambda_{\mathrm{final}}$ [-]", fontsize=12)
ax_fin.set_title(r"Final Steady-State $\lambda$ vs. $\phi$", fontsize=13, fontweight="bold")
ax_fin.grid(True, linestyle="--", alpha=0.6)
ax_fin.legend()

fig2.savefig("results/fig3b_autorotation_stability.png", dpi=300, bbox_inches="tight")
plt.show()
