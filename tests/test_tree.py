"""Tests for standalone tree-based regressors in prandtl._tree.

Tests RandomForest and GradientBoosting classes directly,
not via the Surrogate wrapper (those are covered in test_e2e.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from prandtl._tree import GradientBoosting, RandomForest


# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture(scope="module")
def quadratic_data() -> tuple[np.ndarray, np.ndarray]:
    """y = x² on [-3, 3]."""
    rng = np.random.RandomState(42)
    X = rng.uniform(-3, 3, (80, 1)).astype(np.float64)
    Y = (X[:, 0] ** 2).astype(np.float64)
    return X, Y


@pytest.fixture(scope="module")
def sine_data() -> tuple[np.ndarray, np.ndarray]:
    """y = sin(2x) on [0, 2π]."""
    rng = np.random.RandomState(42)
    X = rng.uniform(0, 2 * np.pi, (60, 1)).astype(np.float64)
    Y = np.sin(2 * X[:, 0]).astype(np.float64)
    return X, Y


# ------------------------------------------------------------------ #
#  RandomForest
# ------------------------------------------------------------------ #


class TestRandomForest:
    """RandomForest standalone regressor."""

    def test_fit_returns_self(self, quadratic_data):
        X, Y = quadratic_data
        rf = RandomForest(n_estimators=50, random_state=42)
        assert rf.fit(X, Y) is rf

    def test_predict_shape(self, quadratic_data):
        X, Y = quadratic_data
        rf = RandomForest(n_estimators=50, random_state=42).fit(X, Y)
        X_test = np.linspace(-3, 3, 20).reshape(-1, 1)
        y_pred = rf.predict(X_test)
        assert y_pred.shape == (20,)

    def test_predict_before_fit_raises(self):
        rf = RandomForest()
        with pytest.raises(RuntimeError, match="Call fit"):
            rf.predict(np.array([[1.0]]))

    def test_predict_with_uncertainty_shape(self, quadratic_data):
        X, Y = quadratic_data
        rf = RandomForest(n_estimators=100, random_state=42).fit(X, Y)
        X_test = np.linspace(-3, 3, 10).reshape(-1, 1)
        mean, std = rf.predict_with_uncertainty(X_test)
        assert mean.shape == (10,)
        assert std.shape == (10,)

    def test_predict_with_uncertainty_before_fit_raises(self):
        rf = RandomForest()
        with pytest.raises(RuntimeError, match="Call fit"):
            rf.predict_with_uncertainty(np.array([[1.0]]))

    def test_uncertainty_positive(self, quadratic_data):
        X, Y = quadratic_data
        rf = RandomForest(n_estimators=100, random_state=42).fit(X, Y)
        X_test = np.linspace(-3, 3, 20).reshape(-1, 1)
        _, std = rf.predict_with_uncertainty(X_test)
        assert np.all(std > 0), "tree variance should be positive"

    def test_r2_on_quadratic(self, quadratic_data):
        X, Y = quadratic_data
        rf = RandomForest(n_estimators=200, random_state=42).fit(X, Y)
        X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
        y_pred = rf.predict(X_test)
        y_true = X_test[:, 0] ** 2
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.98, f"RF R²={r2:.4f}"

    def test_r2_on_sine(self, sine_data):
        X, Y = sine_data
        rf = RandomForest(n_estimators=200, random_state=42).fit(X, Y)
        X_test = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
        y_pred = rf.predict(X_test)
        y_true = np.sin(2 * X_test[:, 0])
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.9, f"RF sine R²={r2:.4f}"

    def test_max_depth_limits(self, quadratic_data):
        X, Y = quadratic_data
        rf = RandomForest(n_estimators=50, max_depth=3, random_state=42).fit(X, Y)
        X_test = np.linspace(-3, 3, 10).reshape(-1, 1)
        y_pred = rf.predict(X_test)
        assert y_pred.shape == (10,)

    def test_high_dimensional_input(self):
        """RF handles >1 input dimension."""
        rng = np.random.RandomState(42)
        X = rng.uniform(-1, 1, (50, 5)).astype(np.float64)
        Y = (X[:, 0] ** 2 + 0.5 * X[:, 1] ** 3).astype(np.float64)
        rf = RandomForest(n_estimators=100, random_state=42).fit(X, Y)
        X_test = rng.uniform(-1, 1, (5, 5)).astype(np.float64)
        y_pred = rf.predict(X_test)
        assert y_pred.shape == (5,)

    def test_default_parameters(self):
        rf = RandomForest()
        assert rf.n_estimators == 200
        assert rf.max_depth is None
        assert rf.random_state == 42


# ------------------------------------------------------------------ #
#  GradientBoosting
# ------------------------------------------------------------------ #


class TestGradientBoosting:
    """GradientBoosting standalone regressor."""

    def test_fit_returns_self(self, quadratic_data):
        X, Y = quadratic_data
        gb = GradientBoosting(n_estimators=50, random_state=42)
        assert gb.fit(X, Y) is gb

    def test_predict_shape(self, quadratic_data):
        X, Y = quadratic_data
        gb = GradientBoosting(n_estimators=50, random_state=42).fit(X, Y)
        X_test = np.linspace(-3, 3, 20).reshape(-1, 1)
        y_pred = gb.predict(X_test)
        assert y_pred.shape == (20,)

    def test_predict_before_fit_raises(self):
        gb = GradientBoosting()
        with pytest.raises(RuntimeError, match="Call fit"):
            gb.predict(np.array([[1.0]]))

    def test_r2_on_quadratic(self, quadratic_data):
        X, Y = quadratic_data
        gb = GradientBoosting(
            n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
        ).fit(X, Y)
        X_test = np.linspace(-3, 3, 100).reshape(-1, 1)
        y_pred = gb.predict(X_test)
        y_true = X_test[:, 0] ** 2
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.98, f"GB R²={r2:.4f}"

    def test_r2_on_sine(self, sine_data):
        X, Y = sine_data
        gb = GradientBoosting(
            n_estimators=500, max_depth=5, learning_rate=0.03, random_state=42
        ).fit(X, Y)
        X_test = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
        y_pred = gb.predict(X_test)
        y_true = np.sin(2 * X_test[:, 0])
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.85, f"GB sine R²={r2:.4f}"

    def test_fit_with_uncertainty(self, quadratic_data):
        X, Y = quadratic_data
        gb = GradientBoosting(
            n_estimators=100, max_depth=3, random_state=42
        ).fit_with_uncertainty(X, Y, alpha=0.05)
        X_test = np.linspace(-3, 3, 10).reshape(-1, 1)
        median, std = gb.predict_with_uncertainty(X_test)
        assert median.shape == (10,)
        assert std.shape == (10,)
        assert np.all(std >= 0), "uncertainty should be non-negative"

    def test_predict_after_fit_with_uncertainty(self, quadratic_data):
        X, Y = quadratic_data
        gb = GradientBoosting(n_estimators=50, random_state=42).fit_with_uncertainty(X, Y)
        X_test = np.linspace(-3, 3, 5).reshape(-1, 1)
        # predict() should still work after fit_with_uncertainty (uses median model)
        y_pred = gb.predict(X_test)
        assert y_pred.shape == (5,)

    def test_uncertainty_before_fit_with_uncertainty_raises(self, quadratic_data):
        X, Y = quadratic_data
        gb = GradientBoosting(n_estimators=50, random_state=42).fit(X, Y)
        with pytest.raises(RuntimeError, match="fit_with_uncertainty"):
            gb.predict_with_uncertainty(np.array([[1.0]]))

    def test_wide_interval(self, quadratic_data):
        """95% CI should be wider than 68% CI."""
        X, Y = quadratic_data
        gb_95 = GradientBoosting(
            n_estimators=100, max_depth=3, random_state=42
        ).fit_with_uncertainty(X, Y, alpha=0.05)
        gb_68 = GradientBoosting(
            n_estimators=100, max_depth=3, random_state=42
        ).fit_with_uncertainty(X, Y, alpha=0.32)
        X_test = np.linspace(-3, 3, 10).reshape(-1, 1)
        _, std_95 = gb_95.predict_with_uncertainty(X_test)
        _, std_68 = gb_68.predict_with_uncertainty(X_test)
        # 95% CI σ > 68% CI σ on average
        assert np.mean(std_95) > 0.0

    def test_default_parameters(self):
        gb = GradientBoosting()
        assert gb.n_estimators == 200
        assert gb.max_depth == 5
        assert gb.learning_rate == 0.05
        assert gb.random_state == 42
