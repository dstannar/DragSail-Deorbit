# compute deorbit

import numpy as np
import matplotlib.pyplot as plt

def compute_deorbit(
    t,
    delta_t,
    state,
    muE=398600,
    Re_km=6378,
    J2=1.08262668e-3,
    Cd=2.2,
    area_m2,
    mass_kg,
    omegaE_rad_s=7.2921150e-5,
    Cr=1.3,
    r_sun_km=None,
):
    """
    State derivative for deorbit propagation.

    Parameters
    ----------
    t : float
        Time [s]
    delta_t : float
        Time Step [s]
    state : ndarray, shape (6,)
        [x, y, z, vx, vy, vz] in [km] and [km/s]
        

    Returns
    -------
    dstate : ndarray, shape (6,)
        [vx, vy, vz, ax, ay, az] in [km/s] and [km/s^2] (probably this needs to be confirmed)
    """
    
     state = np.asarray(state, dtype=float).reshape(6,)
    r = state[:3]
    v = state[3:]

    rmag = np.linalg.norm(r)
    
    t_list = [t]
    alt_list = [rmag]
    t0 = t

    while (rmag > Re_km):

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

        r += v*delta_t + 0.5*a_total*(delta_t**2)
        rmag = np.linalg.norm(r)
        v += a_total*delta_t

        t += delta_t
        t_list.append(t)
        alt_list.append(rmag)

    plt.plot(t_list, alt_list)

    plt.title("Deorbit Simulation")
    plt.xlabel("Time [s]")
    plt.ylabel("Altitude [km]")

    plt.show()

    return t - t0
