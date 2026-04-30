import numpy as np

from datetime import datetime, timedelta, timezone
from nrlmsise00.dataset import msise_4d
from nrlmsise00 import msise_model

from astropy import coordinates as coord
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import CartesianRepresentation, GCRS, ITRS

import geomag_model as gmm



def msis_density_kg_m3(t_dt, alt_km: float, lat_deg: float, lon_deg: float, SPACE_WEATHER) -> float:
    """
    Compute density from NRLMSISE-00 core model, returning g/cm^3.
    """
    f107a, f107, ap = gmm.lookup_space_weather(SPACE_WEATHER, t_dt)

    # gtd7d variant through msise_model wrapper.
    # The Python wrapper typically returns densities in g/cm^3 for total mass density.
    # output[0][5] is usually total mass density.
    output = msise_model(
        t_dt,
        alt_km,
        lat_deg,
        lon_deg,
        f107a,
        f107,
        ap,
    )

    rho_g_cm3 = output[0][5]
    return float(rho_g_cm3)


def drag_acceleration(
    t0,
    t,
    r_eci_km: np.ndarray,
    v_eci_km_s: np.ndarray,
    Cd: float,
    area_m2: float,
    mass_kg: float,
    Re_km: float = 6378.0,
    omegaE_rad_s: float = 7.2921150e-5,
     SPACE_WEATHER=None
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

    t_dt = (t0 + t * u.s).to_datetime(timezone=timezone.utc)

    t_astro = Time(t_dt)

    start_dt = t0.to_datetime(timezone=timezone.utc)
    
    x, y, z = r_eci_km[0], r_eci_km[1], r_eci_km[2]
    cartrep = CartesianRepresentation(x=x*u.km, y=y*u.km, z=z*u.km)
    gcrs = GCRS(cartrep, obstime=t_astro)

    itrs = gcrs.transform_to(ITRS(obstime=t_astro))

    loc = coord.EarthLocation(*itrs.cartesian.xyz)

    lat = loc.lat.deg
    lon = loc.lon.deg

    rho_g_cm3 = msis_density_kg_m3(t_dt, alt_km, lat, lon, SPACE_WEATHER)

    # msise = msise_model(t_dt, alt_km, lat, lon, f107a, f107, ap) # Values for f107a, f107, and ap are semi-arbitrarily chosen

    rho_kg_km3 = rho_g_cm3 * 1e12 # g/cm^3 -> kg/km^3

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
