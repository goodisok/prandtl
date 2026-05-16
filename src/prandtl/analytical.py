"""Analytical truth functions for surrogate model validation."""

from prandtl._analytical import (
    cd_cylinder,
    cl_flat_plate,
    naca0012,
    rae2822,
    thrust_propeller,
)

__all__ = [
    "cl_flat_plate",
    "cd_cylinder",
    "thrust_propeller",
    "naca0012",
    "rae2822",
]
