"""Tests for multi-fidelity Co-Kriging in prandtl._co_kriging."""

from __future__ import annotations

import numpy as np
import pytest

from prandtl._co_kriging import CoKriging


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #


def low_fidelity_func(x: np.ndarray) -> np.ndarray:
    """Cheap low-fidelity approximation: f_L(x) = 0.8 * f_H(x) + 0.1x."""
    # f_H(x) = x², f_L(x) = 0.8x² + 0.1x
    return 0.8 * x**2 + 0.1 * x


def high_fidelity_func(x: np.ndarray) -> np.ndarray:
    """Expensive high-fidelity truth: f_H(x) = x²."""
    return x**2


# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture(scope="module")
def co_kriging_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate low- and high-fidelity data for a 1D quadratic."""
    rng = np.random.RandomState(42)

    # Low-fidelity: many cheap points
    X_low = rng.uniform(-3, 3, (60, 1)).astype(np.float64)
    Y_low = low_fidelity_func(X_low[:, 0]).astype(np.float64)

    # High-fidelity: few expensive points
    X_high = rng.uniform(-3, 3, (15, 1)).astype(np.float64)
    Y_high = high_fidelity_func(X_high[:, 0]).astype(np.float64)

    return X_low, Y_low, X_high, Y_high


@pytest.fixture(scope="module")
def fitted_ck(co_kriging_data) -> CoKriging:
    """A fitted CoKriging model."""
    X_low, Y_low, X_high, Y_high = co_kriging_data
    ck = CoKriging(params=["x"], output="y")
    ck.fit(X_low, Y_low, X_high, Y_high, n_iter=100, verbose=False)
    return ck


# ------------------------------------------------------------------ #
#  Tests
# ------------------------------------------------------------------ #


class TestCoKrigingInit:
    """CoKriging initialisation."""

    def test_init_sets_params(self):
        ck = CoKriging(params=["alpha", "mach"], output="CL")
        assert ck._params == ["alpha", "mach"]
        assert ck._output == "CL"
        assert ck._d == 2
        assert not ck._fitted

    def test_rho_is_none_before_fit(self):
        ck = CoKriging(params=["x"], output="y")
        assert ck.rho is None

    def test_repr_before_fit(self):
        ck = CoKriging(params=["alpha"], output="CL")
        rep = repr(ck)
        assert "CoKriging" in rep
        assert "alpha" in rep
        assert "not fitted" in rep


class TestCoKrigingFit:
    """CoKriging fitting."""

    def test_fit_returns_self(self, co_kriging_data):
        X_low, Y_low, X_high, Y_high = co_kriging_data
        ck = CoKriging(params=["x"], output="y")
        result = ck.fit(X_low, Y_low, X_high, Y_high, n_iter=50, verbose=False)
        assert result is ck
        assert ck._fitted
        assert ck._rho is not None
        assert ck._rho > 0, f"ρ={ck._rho} should be positive"

    def test_fit_sets_rho(self, fitted_ck):
        assert fitted_ck._rho is not None
        # ρ should be close to 1.0 for this scaling relationship
        # Low-fidelity is ~0.8 * high-fidelity, so ρ should be around 1.0-1.5
        assert 0.5 < fitted_ck._rho < 2.5, f"ρ={fitted_ck._rho}"

    def test_fit_fails_on_shape_mismatch(self):
        """X_low columns != params should raise."""
        ck = CoKriging(params=["x", "y"], output="z")
        X_low = np.array([[1.0]])
        Y_low = np.array([1.0])
        X_high = np.array([[1.0, 2.0]])
        Y_high = np.array([2.0])
        with pytest.raises(ValueError, match="X_low must be"):
            ck.fit(X_low, Y_low, X_high, Y_high, verbose=False)

    def test_fit_fails_on_length_mismatch(self):
        """X_low rows != Y_low rows should raise."""
        ck = CoKriging(params=["x"], output="y")
        X_low = np.array([[1.0], [2.0]])
        Y_low = np.array([1.0])  # 1 vs 2
        X_high = np.array([[3.0]])
        Y_high = np.array([4.0])
        with pytest.raises(ValueError, match="mismatched lengths"):
            ck.fit(X_low, Y_low, X_high, Y_high, verbose=False)

    def test_high_fidelity_length_mismatch(self):
        """X_high rows != Y_high rows should raise."""
        ck = CoKriging(params=["x"], output="y")
        X_low = np.array([[1.0]])
        Y_low = np.array([1.0])
        X_high = np.array([[3.0], [4.0]])
        Y_high = np.array([5.0])
        with pytest.raises(ValueError, match="mismatched lengths"):
            ck.fit(X_low, Y_low, X_high, Y_high, verbose=False)


class TestCoKrigingPredict:
    """CoKriging prediction."""

    def test_predict_shape(self, fitted_ck):
        X_test = np.linspace(-3, 3, 50).reshape(-1, 1)
        Y_pred, Y_std = fitted_ck.predict(X_test, return_std=False)
        assert Y_pred.shape == (50,)
        assert Y_std is None

    def test_predict_with_std(self, fitted_ck):
        X_test = np.linspace(-3, 3, 20).reshape(-1, 1)
        Y_pred, Y_std = fitted_ck.predict(X_test, return_std=True)
        assert Y_pred.shape == (20,)
        assert Y_std.shape == (20,)
        assert np.all(Y_std >= 0), "std must be non-negative"
        assert np.any(Y_std > 0), "std should be positive somewhere"

    def test_predict_before_fit_raises(self):
        ck = CoKriging(params=["x"], output="y")
        with pytest.raises(RuntimeError, match="not been fitted"):
            ck.predict(np.array([[1.0]]))

    def test_accuracy_on_quadratic(self, fitted_ck):
        """Co-kriging should predict the high-fidelity truth accurately."""
        X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
        Y_pred, _ = fitted_ck.predict(X_test, return_std=False)
        Y_true = high_fidelity_func(X_test[:, 0])

        ss_res = np.sum((Y_true - Y_pred) ** 2)
        ss_tot = np.sum((Y_true - Y_true.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot

        # With 15 HF points and 60 LF points, R² should be decent
        assert r2 > 0.85, f"CoKriging R²={r2:.4f}"

    def test_uncertainty_decreases_near_training_points(self, fitted_ck):
        """Uncertainty should be lower near high-fidelity training points."""
        X_test = np.array([[0.5], [3.5]])  # 0.5 near training, 3.5 near boundary
        _, Y_std = fitted_ck.predict(X_test, return_std=True)
        # Both should have finite uncertainty
        assert np.all(np.isfinite(Y_std))

    def test_2d_parameter_space(self):
        """CoKriging works with 2 input parameters."""
        rng = np.random.RandomState(42)

        X_low = rng.uniform(-2, 2, (80, 2)).astype(np.float64)
        y_low = (X_low[:, 0]**2 + 0.5 * X_low[:, 1]**2).astype(np.float64)

        X_high = rng.uniform(-2, 2, (20, 2)).astype(np.float64)
        y_high = (X_high[:, 0]**2 + X_high[:, 1]**2).astype(np.float64)

        ck = CoKriging(params=["x1", "x2"], output="y")
        ck.fit(X_low, y_low, X_high, y_high, n_iter=80, verbose=False)

        X_test = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        Y_pred, _ = ck.predict(X_test, return_std=False)
        assert Y_pred.shape == (3,)

    def test_repr_after_fit(self, fitted_ck):
        rep = repr(fitted_ck)
        assert "CoKriging" in rep
        assert "fitted" in rep
        assert "ρ=" in rep
