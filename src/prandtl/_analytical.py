"""Analytical truth functions for framework validation.

These functions provide exact mathematical relationships that serve as
ground-truth targets for testing surrogate model fitting — no CFD data needed.

Each function accepts keyword arguments and returns a dict of output values.
"""

from __future__ import annotations

import numpy as np


def cl_flat_plate(alpha: float, camber: float) -> dict[str, float]:
    """Lift coefficient of a thin flat plate with camber (thin airfoil theory).

    Parameters
    ----------
    alpha : float
        Angle of attack in degrees.
    camber : float
        Camber ratio (e.g., 0.04 for 4% camber), as a fraction of chord.

    Returns
    -------
    dict
        ``{"CL": lift_coefficient}``

    Notes
    -----
    Classic thin airfoil theory: :math:`C_L = 2\\pi (\\alpha + 2c)`,
    where :math:`\\alpha` is angle of attack in radians and :math:`c` is camber ratio.
    Valid for small angles before stall (~ -5° to 15°).
    """
    alpha_rad = np.radians(alpha)
    cl = 2.0 * np.pi * (alpha_rad + 2.0 * camber)
    return {"CL": float(cl)}


def cd_cylinder(reynolds: float) -> dict[str, float]:
    """Drag coefficient of a smooth circular cylinder as a function of Reynolds number.

    Uses a piecewise empirical fit adapted from White (2006) and Clift-Gauvin
    correlations for the subcritical-to-supercritical transition.

    Parameters
    ----------
    reynolds : float
        Reynolds number based on cylinder diameter. Range: 1e-1 to 1e7.

    Returns
    -------
    dict
        ``{"CD": drag_coefficient}``
    """
    Re = reynolds

    if Re < 1.0:
        cd = 24.0 / Re * (1.0 + 0.15 * Re**0.687)
    elif Re < 1e3:
        cd = 1.0
    elif Re < 1e5:
        cd = 1.0 + 0.1 * np.log10(Re / 1e3)
    elif Re < 3e5:
        # Critical transition — rapid drop
        cd = 1.2 - 0.9 * (Re - 1e5) / 2e5
    elif Re < 1e6:
        cd = 0.3 + 0.2 * (1e6 - Re) / 7e5
    else:
        cd = 0.3

    return {"CD": float(cd)}


def thrust_propeller(rpm: float, diameter: float, pitch: float) -> dict[str, float]:
    """Static thrust of a propeller using momentum theory approximation.

    Parameters
    ----------
    rpm : float
        Revolutions per minute.
    diameter : float
        Propeller diameter in meters.
    pitch : float
        Propeller pitch in meters (advance per revolution).

    Returns
    -------
    dict
        ``{"T": thrust_in_newtons}``

    Notes
    -----
    Uses :math:`T = C_T \\rho n^2 D^4` with :math:`C_T \\approx 0.08 \\times \\text{pitch}/D`,
    :math:`\\rho = 1.225` kg/m³ (sea level standard), and :math:`n` in rev/s.
    Valid for static (zero forward speed) condition.
    """
    rho = 1.225  # kg/m³
    n = rpm / 60.0  # rev/s
    ct = 0.08 * pitch / diameter
    thrust = ct * rho * n**2 * diameter**4
    return {"T": float(thrust)}


def naca0012(alpha: float) -> dict[str, float]:
    """Lift and drag coefficients for a NACA 0012 airfoil.

    Uses thin airfoil theory with a smooth stall model for lift, and a
    parabolic drag polar for drag. Valid for incompressible, low-speed flow
    (Mach < 0.3).

    Parameters
    ----------
    alpha : float
        Angle of attack in degrees. Range: -10° to 20°.

    Returns
    -------
    dict
        ``{"CL": lift_coefficient, "CD": drag_coefficient}``

    Notes
    -----
    Lift: :math:`C_L = 2\\pi \\sin(\\alpha_{rad})` below stall, with a
    smooth Gaussian roll-off after :math:`\\alpha_{stall} \\approx 14^\\circ`.

    Drag: :math:`C_D = C_{D0} + k \\cdot C_L^2` where :math:`C_{D0}=0.008`
    and :math:`k = 0.05` (typical for NACA 0012 at Re ~ 3×10⁶).
    """
    alpha_rad = np.radians(alpha)
    alpha_stall_rad = np.radians(14.0)  # stall angle for NACA 0012

    # Smooth stall model: CL ramps up linearly, then rolls off
    cl_linear = 2.0 * np.pi * np.sin(alpha_rad)

    # Gaussian decay factor beyond stall
    cl = np.where(
        np.abs(alpha) < 14.0,
        cl_linear,
        cl_linear * np.exp(-0.5 * ((np.abs(alpha) - 14.0) / 3.0) ** 2),
    )

    # Parabolic drag polar
    cd0 = 0.008
    k = 0.05
    cd = cd0 + k * cl**2

    return {"CL": float(cl), "CD": float(cd)}


def rae2822(alpha: float, mach: float) -> dict[str, float]:
    """Lift and drag coefficients for a RAE 2822 supercritical airfoil.

    Includes Prandtl-Glauert compressibility correction for lift and a
    shock-induced wave drag model for transonic speeds. RAE 2822 is a
    11% thick supercritical airfoil designed for M ≈ 0.7–0.75 cruise.

    Parameters
    ----------
    alpha : float
        Angle of attack in degrees. Range: -5° to 10°.
    mach : float
        Freestream Mach number. Range: 0.1 to 0.85.

    Returns
    -------
    dict
        ``{"CL": lift_coefficient, "CD": drag_coefficient}``

    Notes
    -----
    Lift: :math:`C_L = C_{L0} + C_{L\\alpha} \\cdot \\alpha_{rad}`, then
    Prandtl-Glauert corrected: :math:`C_{L,M} = C_L / \\sqrt{1 - M^2}`.
    :math:`C_{L0} \\approx 0.15` (camber lift at α=0),
    :math:`C_{L\\alpha} \\approx 2\\pi` (thin airfoil lift slope).

    Drag: :math:`C_D = C_{D0} + k C_L^2 + C_{D,wave}(M)`, where
    :math:`C_{D,wave}` is a smooth ramp above the critical Mach number
    (:math:`M_{crit} \\approx 0.73` for RAE 2822).

    References
    ----------
    Cook, P. H., McDonald, M. A., & Firmin, M. C. P. (1979).
    *Aerofoil RAE 2822 — Pressure Distributions, and Boundary Layer and
    Wake Measurements.* AGARD AR-138.
    """
    alpha_rad = np.radians(alpha)

    # Guard against transonic singularity
    mach = np.clip(mach, 0.0, 0.95)
    beta = np.sqrt(np.maximum(1.0 - mach**2, 1e-6))

    # Incompressible lift
    cl0 = 0.15  # camber lift at zero alpha
    cl_alpha = 2.0 * np.pi  # thin airfoil lift slope
    cl_incomp = cl0 + cl_alpha * alpha_rad

    # Prandtl-Glauert compressibility correction
    cl = cl_incomp / beta

    # Drag components
    cd0 = 0.0085  # minimum drag coefficient
    k = 0.045  # induced drag factor

    # Wave drag: smooth ramp above Mcrit ≈ 0.73
    mcrit = 0.73
    cd_wave = np.where(
        mach > mcrit,
        0.02 * ((mach - mcrit) / (0.85 - mcrit)) ** 3,
        0.0,
    )

    cd = cd0 + k * cl**2 + cd_wave

    return {"CL": float(cl), "CD": float(cd)}
