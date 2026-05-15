"""Physics-informed constraints for surrogate model training.

Adds soft penalty terms to the MLP training loss to enforce known
physical relationships — monotonicity, convexity, boundary values —
without requiring a full PDE solver.

Torch is imported lazily at runtime — the module can be imported
without PyTorch installed. Penalty methods raise a clear error if
called without torch.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _torch():
    """Lazy import; raises clear error if pytorch not installed."""
    try:
        import torch  # type: ignore[import-not-found]

        return torch
    except ImportError:
        raise ImportError(
            "Physics constraints require PyTorch. Install with: pip install prandtl[mlp]"
        ) from None


class _PhysicsConstraint:
    """Base class for physics-informed penalty constraints.

    Subclasses implement ``_penalty_torch(x, y_pred)`` which returns
    a scalar torch loss term. The constraint is summed into the total
    training loss with a user-specified weight.

    Parameters
    ----------
    weight : float
        Multiplier for this constraint's contribution to total loss.
    """

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = weight

    def penalty(self, x, y_pred):
        """Compute the penalty as a torch scalar tensor.

        Parameters
        ----------
        x : torch.Tensor
            Input points, shape (N, D).
        y_pred : torch.Tensor
            Model predictions, shape (N,).

        Returns
        -------
        torch.Tensor
            Scalar penalty value.
        """
        return self.weight * self._penalty_torch(x, y_pred)

    def _penalty_torch(self, x, y_pred):
        raise NotImplementedError


class Monotonicity(_PhysicsConstraint):
    """Enforce monotonic relationship: y increases (or decreases) with one input dimension.

    Penalizes violations of dy/dx_{param} >= 0 (if ``sign=1``) or
    dy/dx_{param} <= 0 (if ``sign=-1``).

    Uses finite-difference gradients of sorted input/output pairs.

    Parameters
    ----------
    param_idx : int
        Index of the input dimension to constrain.
    sign : {1, -1}
        1 = output must increase with this parameter (default).
        -1 = output must decrease with this parameter.
    weight : float
        Penalty multiplier.

    Examples
    --------
    >>> # CL must increase with angle of attack
    >>> Monotonicity(0, sign=1, weight=0.1)
    """

    def __init__(self, param_idx: int, sign: int = 1, weight: float = 1.0) -> None:
        super().__init__(weight=weight)
        if sign not in (1, -1):
            raise ValueError(f"sign must be 1 or -1, got {sign}")
        self.param_idx = param_idx
        self.sign = sign

    def __repr__(self) -> str:
        direction = "increasing" if self.sign == 1 else "decreasing"
        return f"Monotonicity(param={self.param_idx}, {direction}, weight={self.weight})"

    def _penalty_torch(self, x, y_pred):
        t = _torch()
        xi = x[:, self.param_idx]
        sorted_idx = t.argsort(xi)
        xi_sorted = xi[sorted_idx]
        y_sorted = y_pred[sorted_idx]
        dx = xi_sorted[1:] - xi_sorted[:-1]
        dy = y_sorted[1:] - y_sorted[:-1]

        mask = t.abs(dx) > 1e-10
        if not mask.any():
            return t.tensor(0.0, device=x.device)

        dydx = dy[mask] / dx[mask]
        violations = t.relu(-self.sign * dydx)
        return t.mean(violations)


class Convexity(_PhysicsConstraint):
    """Enforce convex relationship: d²y/dx² >= 0 for one input dimension.

    Penalizes concavity violations using finite-difference second derivatives.

    Parameters
    ----------
    param_idx : int
        Index of the input dimension to constrain.
    weight : float
        Penalty multiplier.
    """

    def __init__(self, param_idx: int, weight: float = 1.0) -> None:
        super().__init__(weight=weight)
        self.param_idx = param_idx
        self.sign = 1  # convexity = positive second derivative

    def __repr__(self) -> str:
        return f"Convexity(param={self.param_idx}, weight={self.weight})"

    def _penalty_torch(self, x, y_pred):
        t = _torch()
        xi = x[:, self.param_idx]
        sorted_idx = t.argsort(xi)
        xi_sorted = xi[sorted_idx]
        y_sorted = y_pred[sorted_idx]

        dx = xi_sorted[1:] - xi_sorted[:-1]
        dy = y_sorted[1:] - y_sorted[:-1]

        # Second derivative via central differences on midpoints
        dx2 = (dx[1:] + dx[:-1]) / 2  # mid-step sizes
        d2y = (dy[1:] / dx[1:] - dy[:-1] / dx[:-1]) / dx2

        mask = t.abs(dx2) > 1e-10
        if not mask.any():
            return t.tensor(0.0, device=x.device)

        violations = t.relu(-self.sign * d2y[mask])
        return t.mean(violations)


class BoundaryValue(_PhysicsConstraint):
    """Pin model prediction to known values at specific input points.

    Stores raw (unscaled) points and values. The ``Surrogate.fit`` method
    automatically scales them using the same mean/std as training data.
    For MLP training, boundary predictions are computed directly at
    the boundary points and the MSE penalty is added to the loss.

    Parameters
    ----------
    points : dict
        Mapping of parameter name → value at the boundary point.
        Example: ``{"alpha": 0.0}``.
    values : dict
        Mapping of output name → known value.
        Example: ``{"CL": 0.0}``.
    weight : float
        Penalty multiplier.

    Examples
    --------
    >>> # CL must be 0 at α = 0 for symmetric airfoil
    >>> BoundaryValue({"alpha": 0.0}, {"CL": 0.0}, weight=10.0)
    """

    def __init__(self, points: dict, values: dict, weight: float = 1.0) -> None:
        super().__init__(weight=weight)
        # Accept dict for user convenience; convert to arrays internally
        if isinstance(points, dict):
            self._raw_points = dict(points)
            self._raw_values = dict(values)
            self.points = None  # filled by Surrogate.fit
            self.values = None
            if not set(values.keys()).issubset(set(points.keys()) | set(values.keys())):
                # values and points can share keys or not — only check if dict
                pass
        elif isinstance(points, np.ndarray):
            self._raw_points = points.copy()
            self._raw_values = (
                values.copy() if isinstance(values, np.ndarray) else np.atleast_1d(values)
            )
            self.points = None
            self.values = None
            if self._raw_points.shape[0] != self._raw_values.shape[0]:
                raise ValueError(
                    f"Mismatch: {self._raw_points.shape[0]} boundary points vs "
                    f"{self._raw_values.shape[0]} values"
                )
        else:
            raise TypeError("points must be a dict or numpy array")
        if isinstance(points, dict) and len(points) == 0:
            raise ValueError("points must not be empty")

    def __repr__(self) -> str:
        return (
            f"BoundaryValue(points={self._raw_points}, "
            f"values={self._raw_values}, weight={self.weight})"
        )


class CustomConstraint(_PhysicsConstraint):
    """User-defined physics constraint via a callable.

    Parameters
    ----------
    fn : callable
        Function with signature ``fn(x: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor``
        returning a scalar penalty.
    weight : float
        Penalty multiplier.
    """

    def __init__(self, fn: Callable, weight: float = 1.0) -> None:
        super().__init__(weight=weight)
        if not callable(fn):
            raise TypeError("fn must be callable")
        self._fn = fn

    def __repr__(self) -> str:
        return f"CustomConstraint(fn={self._fn.__name__}, weight={self.weight})"

    def _penalty_torch(self, x, y_pred):
        return self._fn(x, y_pred)
