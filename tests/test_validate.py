"""Tests for prandtl._validate — metrics, residual analysis, cross-validation, learning curves."""

import numpy as np
import pytest

import prandtl as pr
from prandtl._validate import cross_validate, learning_curve, metrics, residual_analysis

# ═══════════════════════════════════════════════════════════════════
#  Test data
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def perfect_fit():
    """Y_true == Y_pred — ideal model."""
    rng = np.random.RandomState(42)
    Y = rng.rand(50, 1)
    return Y.ravel(), Y.ravel()


@pytest.fixture
def noisy_fit():
    """Ground truth with small noise added to predictions."""
    rng = np.random.RandomState(42)
    Y_true = rng.rand(100, 1).ravel()
    Y_pred = Y_true + 0.1 * rng.randn(100)
    return Y_true, Y_pred


@pytest.fixture
def surrogate_data():
    """Flat-plate CL data for integration tests."""
    X, Y = pr.sample(
        pr.analytical.cl_flat_plate,
        bounds=[(-5, 15), (0.01, 0.1)],
        n=80,
        seed=42,
    )
    surrogate = pr.Surrogate(params=["alpha", "camber"], outputs=["CL"], method="gp")
    return surrogate, X, Y


# ═══════════════════════════════════════════════════════════════════
#  metrics()
# ═══════════════════════════════════════════════════════════════════


class TestMetrics:
    def test_perfect_fit(self, perfect_fit):
        Y_true, Y_pred = perfect_fit
        m = metrics(Y_true, Y_pred)
        assert m["r2"] == pytest.approx(1.0, abs=1e-6)
        assert m["rmse"] == pytest.approx(0.0, abs=1e-10)
        assert m["mae"] == pytest.approx(0.0, abs=1e-10)
        assert m["explained_variance"] == pytest.approx(1.0, abs=1e-6)

    def test_noisy_fit(self, noisy_fit):
        Y_true, Y_pred = noisy_fit
        m = metrics(Y_true, Y_pred)
        assert 0 < m["r2"] < 1
        assert m["rmse"] > 0
        assert m["mae"] > 0
        assert m["max_error"] >= m["mae"]
        assert m["explained_variance"] > 0

    def test_all_keys_present(self, noisy_fit):
        Y_true, Y_pred = noisy_fit
        m = metrics(Y_true, Y_pred)
        expected = {
            "r2",
            "rmse",
            "mae",
            "mape",
            "max_error",
            "max_relative_error",
            "explained_variance",
        }
        assert set(m.keys()) == expected

    def test_constant_y_true(self):
        Y_true = np.ones(50)
        Y_pred = Y_true + 0.01
        m = metrics(Y_true, Y_pred)
        assert np.isnan(m["r2"])  # zero variance → undefined R²
        assert m["rmse"] == pytest.approx(0.01, abs=1e-4)

    def test_accepts_1d_and_2d(self):
        """metrics should accept flat and column vectors."""
        Y_true = np.array([1, 2, 3, 4, 5])
        Y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.0])
        m1 = metrics(Y_true, Y_pred)
        m2 = metrics(Y_true.reshape(-1, 1), Y_pred.reshape(-1, 1))
        assert m1["r2"] == pytest.approx(m2["r2"])


# ═══════════════════════════════════════════════════════════════════
#  residual_analysis()
# ═══════════════════════════════════════════════════════════════════


class TestResidualAnalysis:
    def test_perfect_fit(self, perfect_fit):
        Y_true, Y_pred = perfect_fit
        r = residual_analysis(Y_true, Y_pred)
        assert r["mean"] == pytest.approx(0.0, abs=1e-10)
        assert r["std"] == pytest.approx(0.0, abs=1e-10)

    def test_noisy_fit(self, noisy_fit):
        Y_true, Y_pred = noisy_fit
        r = residual_analysis(Y_true, Y_pred)
        assert len(r["residuals"]) == 100
        assert abs(r["mean"]) < 1.0  # should be small
        assert r["std"] > 0
        assert isinstance(r["skewness"], float)
        assert isinstance(r["kurtosis"], float)
        assert isinstance(r["max_abs_residual"], float)
        assert 0 <= r["max_abs_residual_index"] < 100

    def test_shapiro_normality(self, noisy_fit):
        """Shapiro-Wilk should run for reasonable sample sizes."""
        Y_true, Y_pred = noisy_fit
        r = residual_analysis(Y_true, Y_pred)
        assert 0 < r["shapiro_pvalue"] <= 1.0

    def test_small_sample_skips_shapiro(self):
        """Shapiro-Wilk requires n >= 3; too-small samples get NaN."""
        Y_true = np.array([1.0, 2.0])
        Y_pred = np.array([1.1, 1.9])
        r = residual_analysis(Y_true, Y_pred)
        assert np.isnan(r["shapiro_statistic"])
        assert np.isnan(r["shapiro_pvalue"])

    def test_all_keys_present(self, noisy_fit):
        Y_true, Y_pred = noisy_fit
        r = residual_analysis(Y_true, Y_pred)
        expected = {
            "residuals",
            "mean",
            "std",
            "skewness",
            "kurtosis",
            "shapiro_statistic",
            "shapiro_pvalue",
            "max_abs_residual",
            "max_abs_residual_index",
        }
        assert set(r.keys()) == expected


# ═══════════════════════════════════════════════════════════════════
#  cross_validate()
# ═══════════════════════════════════════════════════════════════════


class TestCrossValidate:
    def test_basic_cv(self, surrogate_data):
        surrogate, X, Y = surrogate_data
        report = cross_validate(surrogate, X, Y, cv=3)
        assert "CL" in report
        assert report["CL"]["r2_mean"] > 0.99
        assert report["CL"]["rmse_mean"] > 0
        assert 0 <= report["CL"]["r2_std"]
        assert len(report["CL"]["fold_results"]) == 3

    def test_mae_computed_correctly(self, surrogate_data):
        surrogate, X, Y = surrogate_data
        report = cross_validate(surrogate, X, Y, cv=3)
        assert report["CL"]["mae_mean"] > 0
        # MAE should be less than or equal to RMSE (always for same data)
        assert report["CL"]["mae_mean"] <= report["CL"]["rmse_mean"] * 1.1

    def test_cv_must_be_le_samples(self):
        surrogate = pr.Surrogate(params=["x"], outputs=["y"], method="mlp")
        X = np.random.RandomState(0).rand(5, 1)
        Y = X**2
        with pytest.raises(ValueError, match="Number of samples"):
            cross_validate(surrogate, X, Y, cv=10)

    def test_x_y_must_match(self):
        surrogate = pr.Surrogate(params=["x"], outputs=["y"], method="mlp")
        X = np.random.RandomState(0).rand(10, 1)
        Y_short = np.random.RandomState(1).rand(5, 1)
        with pytest.raises(ValueError, match="same number of samples"):
            cross_validate(surrogate, X, Y_short)

    def test_shuffle_reproducibility(self, surrogate_data):
        surrogate, X, Y = surrogate_data
        r1 = cross_validate(surrogate, X, Y, cv=5, random_state=42)
        r2 = cross_validate(surrogate, X, Y, cv=5, random_state=42)
        assert r1["CL"]["r2_mean"] == pytest.approx(r2["CL"]["r2_mean"])

    def test_original_surrogate_not_modified(self, surrogate_data):
        surrogate, X, Y = surrogate_data
        original_method = surrogate._method
        original_params = surrogate._params.copy()
        cross_validate(surrogate, X, Y, cv=3)
        assert surrogate._method == original_method
        assert surrogate._params == original_params

    def test_multi_output(self):
        """Cross-validate a surrogate with 2 outputs."""
        rng = np.random.RandomState(0)
        X = rng.rand(60, 2)
        y1 = X[:, 0] ** 2 + X[:, 1]
        y2 = X[:, 0] - X[:, 1] ** 2
        Y = np.column_stack([y1, y2])

        surrogate = pr.Surrogate(params=["p0", "p1"], outputs=["f1", "f2"], method="gp")
        report = cross_validate(surrogate, X, Y, cv=3, random_state=42)
        assert "f1" in report
        assert "f2" in report
        assert len(report["f1"]["fold_results"]) == 3

    def test_mlp_surrogate(self):
        """Cross-validation works with MLP type too."""
        rng = np.random.RandomState(0)
        X = rng.rand(40, 1)
        Y = 3 * X**2 + 0.1 * rng.randn(40, 1)

        surrogate = pr.Surrogate(params=["x"], outputs=["y"], method="mlp")
        report = cross_validate(surrogate, X, Y, cv=3, random_state=42)
        assert report["y"]["r2_mean"] > 0


# ═══════════════════════════════════════════════════════════════════
#  learning_curve()
# ═══════════════════════════════════════════════════════════════════


class TestLearningCurve:
    def test_basic(self, surrogate_data):
        surrogate, X, Y = surrogate_data
        lc = learning_curve(surrogate, X, Y, train_sizes=[10, 20, 40], cv=3)
        assert lc["train_sizes"] == [10, 20, 40]
        assert len(lc["CL"]["r2_mean"]) == 3
        assert len(lc["CL"]["rmse_mean"]) == 3
        # Performance should improve with more data
        assert lc["CL"]["r2_mean"][-1] >= lc["CL"]["r2_mean"][0] - 0.1

    def test_auto_train_sizes(self, surrogate_data):
        surrogate, X, Y = surrogate_data
        lc = learning_curve(surrogate, X, Y, cv=3)
        assert len(lc["train_sizes"]) == 6  # default 6 sizes
        assert max(lc["train_sizes"]) <= len(X)

    def test_too_large_size_is_dropped(self):
        """train_sizes > n_samples are silently dropped."""
        rng = np.random.RandomState(0)
        X = rng.rand(30, 1)
        Y = X**2
        surrogate = pr.Surrogate(params=["x"], outputs=["y"], method="mlp")
        lc = learning_curve(surrogate, X, Y, train_sizes=[5, 15, 50, 100], cv=3)
        assert max(lc["train_sizes"]) <= 30

    def test_std_arrays(self, surrogate_data):
        surrogate, X, Y = surrogate_data
        lc = learning_curve(surrogate, X, Y, train_sizes=[15, 30, 60], cv=3)
        assert lc["CL"]["r2_std"].shape == (3,)
        assert lc["CL"]["rmse_std"].shape == (3,)
