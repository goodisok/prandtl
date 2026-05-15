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

from prandtl._sampling import sample
from prandtl._surrogate import Surrogate
from prandtl import analytical

__version__ = "0.1.0"
__all__ = ["Surrogate", "sample", "analytical"]
