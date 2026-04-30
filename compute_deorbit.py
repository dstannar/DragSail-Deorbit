# compute deorbit

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import drag_model as dm
import srp_model as srpm
import j2_model as jm
import geomag_model as gmm

from astropy.time import Time
from astropy.coordinates import get_sun, get_body, GCRS
import astropy.units as u

from datetime import timedelta, timezone

# define epoch (you choose reference)
t0 = Time("2026-04-26T21:29:00", scale="utc")

SPACE_WEATHER = gmm.build_space_weather_profile(
    start=t0.to_datetime(timezone=timezone.utc),
    years=5,
)

def sun_vector_eci(t_seconds):
    t = t0 + t_seconds * u.s

    sun = get_sun(t).transform_to(GCRS(obstime=t))

    r_sun_km = sun.cartesian.xyz.to(u.km).value
    return r_sun_km

def moon_vector_eci(t_seconds):
    t = t0 + t_seconds * u.s

    moon = get_body("moon", t).transform_to(GCRS(obstime=t))

    r_moon_km = moon.cartesian.xyz.to(u.km).value
    return r_moon_km

# Pre-calculate Sun and Moon positions
t_sun = np.linspace(0, 5 * 365 * 24 * 3600, 2000)  
r_sun_table = np.array([sun_vector_eci(ti) for ti in t_sun])

t_moon = np.linspace(0, 5 * 365 * 24 * 3600, 20000)
r_moon_table = np.array([moon_vector_eci(ti) for ti in t_moon])

def interp_sun(t):
    return np.array([
    np.interp(t, t_sun, r_sun_table[:, i]) for i in range(3)
])

def interp_moon(t):
    return np.array([
        np.interp(t, t_moon, r_moon_table[:, i]) for i in range(3)
    ])


def deorbit_rhs(
    t,
    state,
    area_m2,
    mass_kg,
    muE=398600.0,          # km^3/s^2
    muSun=1.3271244e11,    # km^3/s^2
    muMoon=4902.8,         # km^3/s^2
    Re_km=6378.0,
    Cd=2.2,
    Cr=1.3,
    omegaE_rad_s=7.2921150e-5
):
    r = state[:3]
    v = state[3:]

    # print("Currently evaluating time " + str(t))

    # Calculate pure distance and velocity
    rmag = np.linalg.norm(r)
    vmag = np.linalg.norm(v)


    # Two-body gravity
    a_tb = -muE * r / rmag**3

    # Drag
    a_drag = dm.drag_acceleration(
        t0=t0,
        t=t,
        r_eci_km=r,
        v_eci_km_s=v,
        Cd=Cd,
        area_m2=area_m2,
        mass_kg=mass_kg,
        Re_km=Re_km,
        omegaE_rad_s=omegaE_rad_s,
        SPACE_WEATHER=SPACE_WEATHER
    )
    
    # J2
    a_j2 = jm.j2_acceleration(
        r=r,
        muE=muE,
        Re=Re_km,
        J2=1.08262668e-3,
    )

    # Sun and Moon
    r_sun = interp_sun(t)
    rSunMag = np.linalg.norm(r_sun)
    r_moon = interp_moon(t)
    rMoonMag = np.linalg.norm(r_moon)

    a_3b_sun  = muSun  * ( (r_sun  - r)/np.linalg.norm(r_sun  - r)**3 - r_sun /rSunMag**3 )
    a_3b_moon = muMoon * ( (r_moon - r)/np.linalg.norm(r_moon - r)**3 - r_moon/rMoonMag**3 );
    

    # SRP (optional for now)
    a_srp = srpm.srp_acceleration(
        r_sc=r,
        r_sun=interp_sun(t),
        mass=mass_kg,
        area_m2=area_m2,
        Cr=Cr,
        Re=Re_km,
    )

    a_total = a_tb + a_drag + a_srp + a_j2 + a_3b_moon + a_3b_sun

    return np.hstack((v, a_total))


def hit_earth_event(t, state, Re_km=6378.0):
    r = state[:3]
    return np.linalg.norm(r) - Re_km

hit_earth_event.terminal = True
hit_earth_event.direction = -1


def compute_deorbit(
    state0,
    area_m2,
    mass_kg,
    t_final=5 * 365 * 24 * 3600,   # 5 years
    n_eval=5000,
    fixed_drag_km_s2=1e-8,
):
    t_span = (0.0, t_final)
    t_eval = np.linspace(t_span[0], t_span[1], n_eval)

    state0 = np.asarray(state0, dtype=float)
    r0mag = np.linalg.norm(state0[:3])
    if r0mag <= 6378.0:
        raise ValueError(f"Initial radius {r0mag:.1f} km is at or below Earth radius.")

    event = lambda t, y, *args: hit_earth_event(t, y, Re_km=6378.0)
    event.terminal = True
    event.direction = -1



    sol = solve_ivp(
        fun=deorbit_rhs,
        t_span=t_span,
        y0=state0,
        args=(area_m2, mass_kg, 398600.0, 1.3271244e11, 4902.8, 6378.0, 2.2, 1.3, 7.2921150e-5),
        t_eval=t_eval,
        events=event,
        rtol=1e-5,
        atol=1e-8,
        method="DOP853",
    )
    
    t_plot = sol.t.copy()
    r_plot = sol.y[:3, :].copy()

    if sol.t_events[0].size > 0:
        t_impact = sol.t_events[0][0]
        state_impact = sol.y_events[0][0]
        r_impact = state_impact[:3].reshape(3, 1)

        # Append only if the event time is not already the last plotted time
        if len(t_plot) == 0 or not np.isclose(t_plot[-1], t_impact):
            t_plot = np.append(t_plot, t_impact)
            r_plot = np.hstack((r_plot, r_impact))

    rmag_plot = np.linalg.norm(r_plot, axis=0)
    altitude_plot = rmag_plot - 6378.0

    plt.plot(t_plot / 86400.0, altitude_plot, linewidth=0.5, color='red')

    r = sol.y[:3, :]
    rmag = np.linalg.norm(r, axis=0)
    altitude = rmag - 6378.0

    plt.plot(sol.t / 86400.0, altitude, linewidth=0.5, color='red')
    plt.title("Deorbit Simulation")
    plt.xlabel("Time [days]")
    plt.ylabel("Altitude [km]")
    plt.grid(True)

    if sol.t_events[0].size > 0:
        t_deorbit = sol.t_events[0][0] / 86400.0  # days
        tdelta = timedelta(t_deorbit)
        print(f"Impact: {str(tdelta)}")


        # Get the state at impact from the event
        r_event = sol.y_events[0][0][:3]
        alt_event = np.linalg.norm(r_event) - 6378.0  # should be ~0

         # Plot the point
        plt.plot(t_deorbit, alt_event, 'o')

         # Label it
        plt.annotate(
            f'Deorbit\n{t_deorbit:.2f} d',
            (t_deorbit, alt_event),
            textcoords="offset points",
            xytext=(5, 10),
            ha='left'
        )
    else:
        print("No deorbit within 5 years")

    plt.show()

    return sol



# Test Case (0.03m^2 Area, 5kg Mass): compute_deorbit([6778, 0, 0, 0, 7.67, 0], 0.03, 5)
