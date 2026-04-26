# compute deorbit

import numpy as np

def compute_deorbit(
    t,
    state,
    muE,
    Re_km,
    J2,
    Cd,
    area_m2,
    mass_kg,
    omegaE_rad_s,
    Cr,
    r_sun_km=None,
):
    """
    State derivative for deorbit propagation.

    Parameters
    ----------
    t : float
        Time [s]
    state : ndarray, shape (6,)
        [x, y, z, vx, vy, vz] in km and km/s

    Returns
    -------
    dstate : ndarray, shape (6,)
        [vx, vy, vz, ax, ay, az]
    """
    state = np.asarray(state, dtype=float).reshape(6,)
    r = state[:3]
    v = state[3:]

    rmag = np.linalg.norm(r)

    # Two-body gravity
    a_tb = -muE * r / rmag**3

    # Drag
    a_drag = drag_acceleration(
        r_eci_km=r,
        v_eci_km_s=v,
        Cd=Cd,
        area_m2=area_m2,
        mass_kg=mass_kg,
        Re_km=Re_km,
        omegaE_rad_s=omegaE_rad_s,
    )

    # J2
    a_j2 = j2_acceleration(
        r=r,
        muE=muE,
        Re=Re_km,
        J2=J2,
    )

    # SRP (optional for now)
    if r_sun_km is None:
        a_srp = np.zeros(3)
    else:
        a_srp = srp_acceleration(
            r_sc=r,
            r_sun=r_sun_km,
            mass=mass_kg,
            area_m2=area_m2,
            Cr=Cr,
            Re=Re_km,
        )

    a_total = a_tb + a_drag + a_j2 + a_srp

    return np.hstack((v, a_total))