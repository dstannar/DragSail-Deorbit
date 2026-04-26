import numpy as np


def atmospheric_density_exponential(alt_km: float) -> float:
    """
    Exponential atmospheric density model.
    Returns density in kg/m^3.
    """
    h = np.array([
        0, 25, 30, 40, 50, 60, 70,
        80, 90, 100, 110, 120, 130, 140,
        150, 180, 200, 250, 300, 350, 400,
        450, 500, 600, 700, 800, 900, 1000
    ], dtype=float)

    rho_ref = np.array([
        1.225, 3.899e-2, 1.774e-2, 3.972e-3, 1.057e-3, 3.206e-4, 8.770e-5,
        1.905e-5, 3.396e-6, 5.297e-7, 9.661e-8, 2.438e-8, 8.484e-9, 3.845e-9,
        2.070e-9, 5.464e-10, 2.789e-10, 7.248e-11, 1.916e-11, 9.518e-12, 3.725e-12,
        1.585e-12, 6.967e-13, 1.454e-13, 3.614e-14, 1.170e-14, 5.245e-15, 3.019e-15
    ], dtype=float)

    H = np.array([
        7.249, 6.349, 6.682, 7.554, 8.382, 7.714, 6.549,
        5.799, 5.382, 5.877, 7.263, 9.473, 12.636, 16.149,
        22.523, 29.740, 37.105, 45.546, 53.628, 58.515, 60.828,
        63.822, 71.835, 88.667, 124.64, 181.05, 268.00
    ], dtype=float)

    alt_km = float(np.clip(alt_km, 0.0, 1000.0))

    if alt_km >= 1000.0:
        i = len(h) - 2
    else:
        i = np.searchsorted(h, alt_km, side="right") - 1
        i = max(0, min(i, len(h) - 2))

    rho = rho_ref[i] * np.exp(-(alt_km - h[i]) / H[i])
    return float(rho)


def drag_acceleration(
    r_eci_km: np.ndarray,
    v_eci_km_s: np.ndarray,
    Cd: float,
    area_m2: float,
    mass_kg: float,
    Re_km: float = 6378.0,
    omegaE_rad_s: float = 7.2921150e-5,
) -> np.ndarray:
    """
    Compute atmospheric drag acceleration in ECI coordinates.
    Returns drag acceleration in km/s^2.
    """
    r_eci_km = np.asarray(r_eci_km, dtype=float).reshape(3,)
    v_eci_km_s = np.asarray(v_eci_km_s, dtype=float).reshape(3,)

    if mass_kg <= 0:
        raise ValueError("mass_kg must be positive")
    if area_m2 < 0:
        raise ValueError("area_m2 must be nonnegative")
    if Cd < 0:
        raise ValueError("Cd must be nonnegative")

    rmag_km = np.linalg.norm(r_eci_km)
    alt_km = rmag_km - Re_km

    rho_kg_m3 = atmospheric_density_exponential(alt_km)
    rho_kg_km3 = rho_kg_m3 * 1e9

    omega_vec = np.array([0.0, 0.0, omegaE_rad_s], dtype=float)
    v_rel_km_s = v_eci_km_s - np.cross(omega_vec, r_eci_km)

    area_km2 = area_m2 * 1e-6

    v_rel_mag = np.linalg.norm(v_rel_km_s)
    if v_rel_mag < 1e-15:
        return np.zeros(3, dtype=float)

    a_drag_km_s2 = (
        -0.5
        * Cd
        * (area_km2 / mass_kg)
        * rho_kg_km3
        * v_rel_mag
        * v_rel_km_s
    )

    return a_drag_km_s2