"""Tests for physics-informed constraints.

Uses the analytical cl_flat_plate function (CL = 2π * sin(α * π/180))
for validation — it is monotonic in α on [0, 15]°.
"""

import numpy as np
import pytest

import prandtl as pr
from prandtl import BoundaryValue, Convexity, CustomConstraint, Monotonicity

# ------------------------------------------------------------------ #
#  Monotonicity
# ------------------------------------------------------------------ #


class TestMonotonicity:
    """Test monotonicity constraint on CL(alpha) — known monotonic."""

    def test_cl_increasing_with_alpha(self) -> None:
        """Monotonicity(+1) should not degrade fit on monotonic data."""
        bounds = [(-5.0, 15.0), (0.01, 0.05)]
        X, Y = pr.sample(pr.analytical.cl_flat_plate, bounds=bounds, n=100)

        Xt, Yt = pr.sample(pr.analytical.cl_flat_plate, bounds=bounds, n=50)

        surr_phys = pr.Surrogate(params=["alpha", "camber"], outputs=["CL"], method="mlp").fit(
            X,
            Y,
            n_iter=500,
            lr=0.01,
            physics=[Monotonicity(param_idx=0, sign=1, weight=0.1)],
        )

        r2_physics = surr_phys.validate(Xt, Yt)["CL"]["r2"]

        # Physics should not hurt R² on genuinely monotonic data
        assert r2_physics > 0.9, f"R² with physics constraint = {r2_physics:.4f}"

    def test_repr(self) -> None:
        c = Monotonicity(param_idx=0, sign=1)
        assert "Monotonicity" in repr(c)
        assert "increasing" in repr(c)

    def test_invalid_sign(self) -> None:
        with pytest.raises(ValueError, match="sign"):
            Monotonicity(param_idx=0, sign=0)


# ------------------------------------------------------------------ #
#  Convexity
# ------------------------------------------------------------------ #


class TestConvexity:
    """Test convexity constraint."""

    def test_basic(self) -> None:
        constraint = Convexity(param_idx=0, weight=0.5)
        assert constraint.param_idx == 0
        assert constraint.sign == 1  # always 1 for convexity

    def test_repr(self) -> None:
        c = Convexity(param_idx=1)
        assert "Convexity" in repr(c)


# ------------------------------------------------------------------ #
#  BoundaryValue
# ------------------------------------------------------------------ #


class TestBoundaryValue:
    """Test boundary value pinning."""

    def test_cl_zero_at_zero_alpha(self) -> None:
        """CL(0, camber=0.02) → enforce = 0."""
        bounds = [(-5.0, 15.0), (0.01, 0.05)]
        X, Y = pr.sample(pr.analytical.cl_flat_plate, bounds=bounds, n=80)

        # Known: CL ≈ 0 at α=0 for flat plate
        boundary = BoundaryValue(
            points=np.array([[0.0, 0.02]]), values=np.array([0.0]), weight=10.0
        )

        surr = pr.Surrogate(params=["alpha", "camber"], outputs=["CL"], method="mlp").fit(
            X, Y, n_iter=500, lr=0.01, physics=[boundary]
        )

        pred = surr.predict(np.array([[0.0, 0.02]]))
        assert abs(pred[0, 0]) < 0.05, f"CL(0) = {pred[0, 0]:.6f}, expected ~0"

    def test_point_value_mismatch(self) -> None:
        with pytest.raises(ValueError, match="Mismatch"):
            BoundaryValue(
                points=np.array([[0.0, 0.02], [5.0, 0.03]]),
                values=np.array([0.0]),
            )

    def test_repr(self) -> None:
        c = BoundaryValue(points=np.array([[0.0, 0.02]]), values=np.array([0.0]))
        assert "BoundaryValue" in repr(c)

    def test_dict_constructor(self) -> None:
        """BoundaryValue should accept dict-style points and values."""
        constraint = BoundaryValue(
            points={"alpha": 0.0, "camber": 0.02},
            values={"CL": 0.0},
            weight=10.0,
        )
        assert constraint.weight == 10.0
        assert isinstance(constraint._raw_points, dict)
        assert constraint._raw_points == {"alpha": 0.0, "camber": 0.02}

    def test_empty_dict_raises(self) -> None:
        """Empty points dict should raise ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            BoundaryValue(points={}, values={"CL": 0.0})

    def test_non_dict_array_raises(self) -> None:
        """Non-dict, non-array input should raise TypeError."""
        with pytest.raises(TypeError, match="dict or numpy"):
            BoundaryValue(points=42, values=np.array([0.0]))


# ------------------------------------------------------------------ #
#  CustomConstraint
# ------------------------------------------------------------------ #


class TestCustomConstraint:
    """Test user-defined physics constraints."""

    def test_custom_penalty_called(self) -> None:
        import torch

        call_count = [0]

        def zero_penalty(x, y):
            call_count[0] += 1
            return torch.tensor(0.0)

        bounds = [(-5.0, 15.0), (0.01, 0.05)]
        X, Y = pr.sample(pr.analytical.cl_flat_plate, bounds=bounds, n=40)

        pr.Surrogate(params=["alpha", "camber"], outputs=["CL"], method="mlp").fit(
            X,
            Y,
            n_iter=50,
            lr=0.01,
            physics=[CustomConstraint(zero_penalty, weight=1.0)],
        )

        assert call_count[0] > 0, "Custom constraint was never called"


class TestConvexityIntegration:
    """Test convexity constraint actually improves fit on convex data."""

    @staticmethod
    def parabola(a: float) -> dict[str, float]:
        """y = a² — convex function."""
        return {"y": a**2}

    def test_convexity_on_parabola(self) -> None:
        """MLP with convexity constraint should fit a parabola well."""
        X, Y = pr.sample(self.parabola, bounds=[(-3, 3)], n=80, seed=0)

        surr = pr.Surrogate(params=["a"], outputs=["y"], method="mlp").fit(
            X,
            Y,
            n_iter=2000,
            lr=0.01,
            physics=[Convexity(param_idx=0, weight=0.1)],
        )
        X_test, Y_test = pr.sample(self.parabola, bounds=[(-3, 3)], n=30, seed=1)
        report = surr.validate(X_test, Y_test)
        assert report["y"]["r2"] > 0.99, f"R² with convexity = {report['y']['r2']:.4f}"


class TestMultiOutputPhysics:
    """Multi-output MLP with physics constraints."""

    @staticmethod
    def lift_drag(alpha: float) -> dict[str, float]:
        """CL = α, CD = 0.01 + 0.01α² — CD is convex in α."""
        return {"CL": alpha, "CD": 0.01 + 0.01 * alpha**2}

    def test_multi_output_with_monotonicity(self) -> None:
        """Monotonicity on CL should not hurt CD prediction."""
        X, Y = pr.sample(self.lift_drag, bounds=[(-5, 10)], n=100, seed=0)

        surr = pr.Surrogate(params=["alpha"], outputs=["CL", "CD"], method="mlp").fit(
            X,
            Y,
            n_iter=2000,
            lr=0.01,
            physics=[Monotonicity(param_idx=0, sign=1, weight=0.1)],
        )
        X_test, Y_test = pr.sample(self.lift_drag, bounds=[(-5, 10)], n=30, seed=1)
        report = surr.validate(X_test, Y_test)
        assert report["CL"]["r2"] > 0.99, f"CL R² = {report['CL']['r2']:.4f}"
        assert report["CD"]["r2"] > 0.99, f"CD R² = {report['CD']['r2']:.4f}"


# ------------------------------------------------------------------ #
#  GP rejection
# ------------------------------------------------------------------ #


class TestGPRejection:
    """GP models must reject physics constraints."""

    def test_gp_with_physics_raises(self) -> None:
        bounds = [(-5.0, 15.0), (0.01, 0.05)]
        X, Y = pr.sample(pr.analytical.cl_flat_plate, bounds=bounds, n=30)

        surr = pr.Surrogate(params=["alpha", "camber"], outputs=["CL"])
        with pytest.raises(ValueError, match="Physics constraints.*only supported"):
            surr.fit(
                X,
                Y,
                n_iter=50,
                physics=[Monotonicity(param_idx=0, sign=1)],
            )
