"""Prandtl — CFD surrogate modeling toolkit.

Train, validate, and export surrogate models for aerodynamic predictions
with a scikit-learn-like interface.

Quick start::

    import prandtl as pr

    # 1. Sample parameter space with analytical truth
    X, Y = pr.sample(pr.analytical.cl_flat_plate, bounds=[(-5, 15), (0.01, 0.1)], n=100)

    # 2. Fit a surrogate model
    surrogate = pr.Surrogate(params=["alpha", "camber"], outputs=["CL"]).fit(X, Y)

    # 3. Validate
    X_test, Y_test = pr.sample(pr.analytical.cl_flat_plate, bounds=[(-5, 15), (0.01, 0.1)], n=20)
    report = surrogate.validate(X_test, Y_test)
    print(f"R² = {report['CL']['r2']:.4f}")
"""

from prandtl import analytical
from prandtl._active import active_learn, propose_next
from prandtl._co_kriging import CoKriging
from prandtl._io import read_foam_forces, read_su2_history
from prandtl._physics import BoundaryValue, Convexity, CustomConstraint, Monotonicity
from prandtl._sampling import sample
from prandtl._sobolev import soboloev
from prandtl._surrogate import Surrogate
from prandtl._tree import GradientBoosting, RandomForest
from prandtl._validate import cross_validate, learning_curve, metrics, residual_analysis

__version__ = "0.5.1"
__all__ = [
    "Surrogate",
    "CoKriging",
    "active_learn",
    "cross_validate",
    "learning_curve",
    "metrics",
    "propose_next",
    "read_foam_forces",
    "read_su2_history",
    "residual_analysis",
    "sample",
    "soboloev",
    "analytical",
    "RandomForest",
    "GradientBoosting",
    "Monotonicity",
    "Convexity",
    "BoundaryValue",
    "CustomConstraint",
]
