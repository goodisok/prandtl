"""Tests for Sobolev (gradient-constrained) GP training in prandtl._sobolev."""

from __future__ import annotations

import numpy as np
import pytest

from prandtl._sobolev import fit_with_gradients, predict_gradient, soboloev


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #


def sin_1d_with_gradient(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """f(x) = sin(3x), f'(x) = 3*cos(3x)."""
    return np.sin(3 * x), 3 * np.cos(3 * x)


# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture(scope="module")
def soboloev_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1D sin data with gradients."""
    rng = np.random.RandomState(42)
    X = rng.uniform(0, 2, (12, 1)).astype(np.float64)
    Y, dY = sin_1d_with_gradient(X[:, 0])
    return X, Y, dY.reshape(-1, 1)


@pytest.fixture(scope="module")
def fitted_soboloev(soboloev_data) -> soboloev:
    """A fitted soboloev model."""
    X, Y, dY = soboloev_data
    mdl = soboloev(params=["x"], output="y", kernel="rbf", grad_weight=0.5)
    mdl.fit(X, Y, dY, n_iter=200, verbose=False)
    return mdl


# ------------------------------------------------------------------ #
#  soboloev class
# ------------------------------------------------------------------ #


class TestSobolevInit:
    """soboloev initialisation."""

    def test_init_sets_parameters(self):
        mdl = soboloev(params=["alpha"], output="CL", kernel="rbf", grad_weight=0.3)
        assert mdl.params == ["alpha"]
        assert mdl.output == "CL"
        assert mdl.kernel == "rbf"
        assert mdl.grad_weight == 0.3

    def test_invalid_kernel_raises(self):
        with pytest.raises(ValueError, match="kernel"):
            soboloev(params=["x"], output="y", kernel="poly")

    def test_invalid_grad_weight_raises(self):
        with pytest.raises(ValueError, match="grad_weight"):
            soboloev(params=["x"], output="y", grad_weight=1.5)

        with pytest.raises(ValueError, match="grad_weight"):
            soboloev(params=["x"], output="y", grad_weight=-0.1)

    def test_grad_weight_boundary_values(self):
        """0 and 1 are valid boundary values."""
        mdl_zero = soboloev(params=["x"], output="y", grad_weight=0.0)
        mdl_one = soboloev(params=["x"], output="y", grad_weight=1.0)
        assert mdl_zero.grad_weight == 0.0
        assert mdl_one.grad_weight == 1.0

    def test_repr(self):
        mdl = soboloev(params=["x"], output="f")
        rep = repr(mdl)
        assert "soboloev" in rep
        assert "'x'" in rep


class TestSobolevFit:
    """soboloev fitting."""

    def test_fit_returns_self(self, soboloev_data):
        X, Y, dY = soboloev_data
        mdl = soboloev(params=["x"], output="y")
        result = mdl.fit(X, Y, dY, n_iter=50, verbose=False)
        assert result is mdl

    def test_fit_stores_internals(self, soboloev_data):
        X, Y, dY = soboloev_data
        mdl = soboloev(params=["x"], output="y")
        mdl.fit(X, Y, dY, n_iter=50, verbose=False)
        assert mdl._model is not None
        assert mdl._xm is not None
        assert mdl._xs is not None
        assert mdl._ym is not None
        assert mdl._ys is not None

    def test_fit_with_matern52_kernel(self, soboloev_data):
        X, Y, dY = soboloev_data
        mdl = soboloev(params=["x"], output="y", kernel="matern52")
        mdl.fit(X, Y, dY, n_iter=50, verbose=False)
        assert mdl._model is not None

    def test_shape_mismatch_raises(self):
        """X and dY_dX must have same number of samples."""
        X = np.array([[1.0], [2.0]])
        Y = np.array([1.0, 4.0])
        dY = np.array([[2.0]])  # 1 sample, not 2
        mdl = soboloev(params=["x"], output="y")
        with pytest.raises(ValueError, match="same number of samples"):
            mdl.fit(X, Y, dY, n_iter=10, verbose=False)

    def test_dimension_mismatch_raises(self):
        """X and dY_dX must have same input dimension."""
        X = np.array([[1.0]])
        Y = np.array([1.0])
        dY = np.array([[2.0, 3.0]])  # 2 dims, not 1
        mdl = soboloev(params=["x"], output="y")
        with pytest.raises(ValueError, match="same input dimension"):
            mdl.fit(X, Y, dY, n_iter=10, verbose=False)


class TestSobolevPredict:
    """soboloev prediction."""

    def test_predict_shape(self, fitted_soboloev):
        X_test = np.linspace(0, 2, 50).reshape(-1, 1)
        y_pred = fitted_soboloev.predict(X_test)
        assert y_pred.shape == (50,)

    def test_predict_before_fit_raises(self):
        mdl = soboloev(params=["x"], output="y")
        with pytest.raises(RuntimeError, match="Call fit"):
            mdl.predict(np.array([[1.0]]))

    def test_predict_gradient_shape(self, fitted_soboloev):
        X_test = np.linspace(0, 2, 20).reshape(-1, 1)
        grad = fitted_soboloev.predict_gradient(X_test)
        assert grad.shape == (20, 1)

    def test_predict_gradient_before_fit_raises(self):
        mdl = soboloev(params=["x"], output="y")
        with pytest.raises(RuntimeError, match="Call fit"):
            mdl.predict_gradient(np.array([[1.0]]))

    def test_accuracy_on_sine(self, fitted_soboloev):
        """With gradient info, soboloev should fit sin(3x) accurately with only 12 points."""
        X_test = np.linspace(0, 2, 100).reshape(-1, 1)
        y_pred = fitted_soboloev.predict(X_test)
        y_true = np.sin(3 * X_test[:, 0])

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.9, f"sobolev R²={r2:.4f}"

    def test_gradient_accuracy(self, fitted_soboloev):
        """Predicted gradient should approximately match true gradient."""
        X_test = np.linspace(0.1, 1.9, 20).reshape(-1, 1)
        grad_pred = fitted_soboloev.predict_gradient(X_test)
        grad_true = 3 * np.cos(3 * X_test[:, 0])
        # Convert to numpy for compatibility (predict_gradient may return tensor)
        gpred = np.asarray(grad_pred, dtype=np.float64).ravel()
        gtrue = np.asarray(grad_true, dtype=np.float64).ravel()
        # Check correlation: gradients should have same sign pattern
        sign_match = np.mean(np.sign(gpred) == np.sign(gtrue))
        # At least 65% of predictions have same sign as truth
        assert sign_match > 0.65, f"Sign match: {sign_match:.2f}"


# ------------------------------------------------------------------ #
#  fit_with_gradients (standalone function)
# ------------------------------------------------------------------ #


class TestFitWithGradients:
    """fit_with_gradients() standalone function."""

    def test_returns_model_and_stats(self, soboloev_data):
        X, Y, dY = soboloev_data
        model, x_mean, x_std, y_mean, y_std = fit_with_gradients(
            X, Y, dY, n_iter=100, verbose=False
        )
        # Verify return types
        assert model is not None
        assert isinstance(x_mean, np.ndarray)
        assert isinstance(x_std, np.ndarray)
        assert isinstance(y_mean, float)
        assert isinstance(y_std, float)

    def test_gradient_only_training(self):
        """grad_weight=1.0 (gradient-only) converges."""
        rng = np.random.RandomState(42)
        X = rng.uniform(0, 2, (10, 1)).astype(np.float64)
        Y, dY = sin_1d_with_gradient(X[:, 0])
        dY = dY.reshape(-1, 1)

        model, x_mean, x_std, y_mean, y_std = fit_with_gradients(
            X, Y, dY, n_iter=100, grad_weight=1.0, verbose=False
        )
        assert model is not None
        assert np.isfinite(float(y_std))


# ------------------------------------------------------------------ #
#  predict_gradient (standalone function)
# ------------------------------------------------------------------ #


class TestPredictGradient:
    """predict_gradient() standalone function."""

    def test_shape(self, soboloev_data):
        X, Y, dY = soboloev_data
        model, x_mean, x_std, y_mean, y_std = fit_with_gradients(
            X, Y, dY, n_iter=100, verbose=False
        )

        import torch
        x_t = torch.tensor((X - x_mean) / x_std, dtype=torch.float32)
        x_q = torch.tensor((np.array([[0.5], [1.5]]) - x_mean) / x_std, dtype=torch.float32)

        grad = predict_gradient(model, x_t, x_q)
        assert grad.shape == (2, 1)

    def test_gradient_on_training_points(self, soboloev_data):
        """Gradient at training points should approximate true gradient."""
        X, Y, dY = soboloev_data
        model, x_mean, x_std, y_mean, y_std = fit_with_gradients(
            X, Y, dY, n_iter=200, grad_weight=0.8, verbose=False
        )

        import torch
        x_t = torch.tensor((X - x_mean) / x_std, dtype=torch.float32)
        grad_pred = predict_gradient(model, x_t, x_t)  # train_x = query_x
        # Convert to numpy first for compatibility
        gpred = np.asarray(grad_pred * y_std / x_std, dtype=np.float64)
        gtrue = np.asarray(dY, dtype=np.float64)

        # Signs should mostly match
        sign_match = np.mean(np.sign(gpred) == np.sign(gtrue))
        assert sign_match > 0.6, f"Gradient sign match: {sign_match:.2f}"
