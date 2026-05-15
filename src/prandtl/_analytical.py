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
