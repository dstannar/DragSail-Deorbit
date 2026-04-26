# jpert model

function dstate = Cowells_J2_full(t, state, ...
    muE, Re, J2, ...
    Cd, Cr, area_km2, mass, omegaE, ...
    muSun, muMoon, P0, AU_km, jd0)

    r = state(1:3);
    v = state(4:6);

    rmag = norm(r);
    x = r(1);
    y = r(2);
    z = r(3);

    % Two-body gravity
    a_tb = -muE * r / rmag^3;

    % Drag
    omegaVec = [0; 0; omegaE];
    v_rel = v - cross(omegaVec, r);
    alt = rmag - Re;
    rho = ExponDensModData(alt) * 1e9;
    a_drag = -0.5 * Cd * (area_km2/mass) * rho * norm(v_rel) * v_rel;

    % J2 only
    r2 = rmag^2;
    z2 = z^2;
    kJ2 = 1.5 * J2 * muE * Re^2 / rmag^5;

    aJ2x = -kJ2 * x * (1 - 5*z2/r2);
    aJ2y = -kJ2 * y * (1 - 5*z2/r2);
    aJ2z = -kJ2 * z * (3 - 5*z2/r2);

    a_J2 = [aJ2x; aJ2y; aJ2z];

    % Time
    jd = jd0 + t/86400;

    % Sun and Moon
    [~, ~, rSun] = solar_position(jd);
    rSun = rSun(:);
    rSunMag = norm(rSun);

    rMoon = lunar_position(jd);
    rMoon = rMoon(:);
    rMoonMag = norm(rMoon);

    a_3b_sun  = muSun  * ( (rSun  - r)/norm(rSun  - r)^3 - rSun /rSunMag^3 );
    a_3b_moon = muMoon * ( (rMoon - r)/norm(rMoon - r)^3 - rMoon/rMoonMag^3 );

    % SRP
    area_m2 = area_km2 * 1e6;

    r_sc2sun = rSun - r;
    r_sc2sun_mag = norm(r_sc2sun);

    theta  = acos(dot(rSun, r)/(rSunMag*rmag));
    theta1 = acos(Re / rmag);
    theta2 = acos(Re / rSunMag);

    if theta1 + theta2 < theta
        F_eclipse = 0;
    else
        F_eclipse = 1;
    end

    P = P0 * (AU_km / rSunMag)^2;

    a_srp = F_eclipse * (P * Cr * area_m2 / mass) / 1000 ...
            * (-r_sc2sun / r_sc2sun_mag);

    % Total
    a_total = a_tb + a_drag + a_J2 + a_3b_sun + a_3b_moon + a_srp;

    dstate = [v; a_total];
end