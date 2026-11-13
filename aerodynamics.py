"""
Aerodynamic modeling and airfoil polar interpolation module for H-Rotor VAWTs.
Supports direct loading of QBlade/AeroDyn v13.0 (.dat), CSV, and Excel polars.
"""

from pathlib import Path
from typing import Tuple, Callable
import numpy as np
import pandas as pd
from scipy import interpolate


def load_airfoil_polars(
    filepath: str | Path,
    sheet_name: str | int = 0
) -> Tuple[Callable, Callable, Callable]:
    """
    Loads 360-degree airfoil polar data (.dat from QBlade, .csv, or .xlsx)
    and returns cubic spline interpolators for Cl(alpha), Cd(alpha), and Cm(alpha) in radians.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Airfoil polar file not found: {path}")

    ext = path.suffix.lower()
    if ext == ".dat":
        # Reads QBlade / AeroDyn v13.0 format skipping the 14-line metadata header
        df = pd.read_csv(
            path,
            sep=r"\s+",
            skiprows=14,
            names=["alpha", "cl", "cd", "cm"],
            engine="python"
        )
    elif ext == ".csv":
        df = pd.read_csv(path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path, sheet_name=sheet_name)
    else:
        raise ValueError("Unsupported file format. Use .dat, .csv, or .xlsx")

    df.columns = [col.strip().lower() for col in df.columns]
    alpha_col = "alpha" if "alpha" in df.columns else "x"

    df = df.dropna(subset=[alpha_col, "cl", "cd"])
    df = df.sort_values(by=alpha_col).drop_duplicates(subset=[alpha_col])

    alpha_rad = np.deg2rad(df[alpha_col].to_numpy(dtype=float))
    cl = df["cl"].to_numpy(dtype=float)
    cd = np.maximum(df["cd"].to_numpy(dtype=float), 0.0)
    cm = df["cm"].to_numpy(dtype=float) if "cm" in df.columns else np.zeros_like(cl)

    cl_interp = interpolate.interp1d(alpha_rad, cl, kind="cubic", fill_value="extrapolate")
    cd_interp = interpolate.interp1d(alpha_rad, cd, kind="cubic", fill_value="extrapolate")
    cm_interp = interpolate.interp1d(alpha_rad, cm, kind="cubic", fill_value="extrapolate")

    return cl_interp, cd_interp, cm_interp


def dimensionless_relative_velocity(theta: np.ndarray | float, tsr: float) -> np.ndarray | float:
    """Computes dimensionless relative wind speed U_R / U_inf."""
    return np.sqrt(tsr**2 + 2.0 * tsr * np.sin(theta) + 1.0)


def inflow_angle(theta: np.ndarray | float, tsr: float) -> np.ndarray | float:
    """Computes local flow inflow angle gamma [rad] between U_R and blade velocity."""
    return np.arctan2(np.cos(theta), np.sin(theta) + tsr)


def aerodynamic_force_coeffs(
    theta: np.ndarray | float,
    tsr: float,
    phi: np.ndarray | float,
    cl_interp: Callable,
    cd_interp: Callable
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes local normal (Cn), tangential (Ct), and global hub force
    coefficients (Cfx, Cfz) following the TFG formulation.
    """
    gam = inflow_angle(theta, tsr)
    alpha = gam + phi

    cl = cl_interp(alpha)
    cd = cd_interp(alpha)

    cn = cd * np.sin(gam) + cl * np.cos(gam)
    ct = cl * np.sin(gam) - cd * np.cos(gam)

    cfz = cn * np.sin(theta) + ct * np.cos(theta)
    cfx = cn * np.cos(theta) - ct * np.sin(theta)

    return cn, ct, cfx, cfz



def aerodynamic_moment_coeff(
    theta: np.ndarray | float,
    tsr: float,
    phi: np.ndarray | float,
    d_tilde: float,
    c_tilde: float,
    cl_interp: Callable,
    cd_interp: Callable,
    cm_interp: Callable
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes instantaneous power coefficient (Cp) and dimensionless aerodynamic
    torque coefficient (Cm_aero) accounting for pivot eccentricity (d_tilde = d/R)
    and airfoil pitching moment (c_tilde = c/R).
    """
    gam = inflow_angle(theta, tsr)
    alpha = gam + phi
    u_r_tilde = dimensionless_relative_velocity(theta, tsr)

    cl = cl_interp(alpha)
    cd = cd_interp(alpha)
    cm = cm_interp(alpha)

    cp = cl * np.cos(theta) - cd * (np.sin(theta) + tsr)

    eccentricity_term = u_r_tilde * d_tilde * (
        cl * (np.sin(alpha) - tsr * np.cos(phi)) - cd * (np.cos(alpha) + tsr * np.sin(phi))
    )
    pitching_term = -(u_r_tilde**2) * c_tilde * cm

    cm_aero = u_r_tilde * cp + eccentricity_term + pitching_term
    return cp, cm_aero
