#srp model and solar position

# function takes time, state, mu, F, rsun, rsunmag

import numpy as np

def srp_acceleration(r_sc, r_sun, mass, area_m2, Cr,
                     P0=4.56e-6, AU_km=149597870.691,
                     eclipse=True, Re=6378.0):
    """
    Compute Solar Radiation Pressure (SRP) acceleration.

    Parameters
    ----------
    r_sc : ndarray (3,)
        Spacecraft position vector in ECI [km]

    r_sun : ndarray (3,)
        Sun position vector in ECI (Earth -> Sun) [km]

    mass : float
        Spacecraft mass [kg]

    area_m2 : float
        Effective cross-sectional area [m^2]

    Cr : float
        Reflectivity coefficient

    P0 : float, optional
        Solar radiation pressure at 1 AU [N/m^2]

    AU_km : float, optional
        Astronomical Unit [km]

    eclipse : bool, optional
        Whether to apply Earth shadow model

    Re : float, optional
        Earth radius [km]

    Returns
    -------
    a_srp : ndarray (3,)
        SRP acceleration vector [km/s^2]
    """

    r_sc = np.asarray(r_sc, dtype=float)
    r_sun = np.asarray(r_sun, dtype=float)

    rmag = np.linalg.norm(r_sc)
    rSunMag = np.linalg.norm(r_sun)

    # --- spacecraft to Sun vector ---
    r_sc2sun = r_sun - r_sc
    r_sc2sun_mag = np.linalg.norm(r_sc2sun)

    # --- eclipse factor ---
    F_eclipse = 1.0

    if eclipse:
        # angle between Earth->Sun and Earth->spacecraft
        cos_theta = np.dot(r_sun, r_sc) / (rSunMag * rmag)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)

        theta1 = np.arccos(Re / rmag)
        theta2 = np.arccos(Re / rSunMag)

        if theta1 + theta2 < theta:
            F_eclipse = 0.0

    # --- SRP pressure at current distance ---
    P = P0 * (AU_km / r_sc2sun_mag)**2  # N/m^2

    # --- SRP acceleration (convert m/s^2 → km/s^2) ---
    a_srp = F_eclipse * (P * Cr * area_m2 / mass) / 1000.0 \
            * (-r_sc2sun / r_sc2sun_mag)

    return a_srp
