"""End-to-end validation: analytical truth → surrogate → metrics.

Run with:
    pytest tests/test_e2e.py -v -s
"""

from __future__ import annotations

import numpy as np
import pytest

import prandtl as pr

# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture(scope="module")
def gp_surrogate():
    """Train a GP surrogate on thin airfoil lift coefficient data."""
    X, Y = pr.sample(
        pr.analytical.cl_flat_plate,
        bounds=[(-5, 15), (0.01, 0.1)],
        n=100,
        seed=42,
    )
    surrogate = pr.Surrogate(params=["alpha", "camber"], outputs=["CL"], method="gp")
    surrogate.fit(X, Y, verbose=False)
    return surrogate, X, Y


@pytest.fixture(scope="module")
def mlp_surrogate():
    """Train an MLP surrogate on thin airfoil lift coefficient data."""
    X, Y = pr.sample(
        pr.analytical.cl_flat_plate,
        bounds=[(-5, 15), (0.01, 0.1)],
        n=100,
        seed=42,
    )
    surrogate = pr.Surrogate(params=["alpha", "camber"], outputs=["CL"], method="mlp")
    surrogate.fit(X, Y, n_iter=2000, verbose=False)
    return surrogate, X, Y


# ------------------------------------------------------------------ #
#  Tests
# ------------------------------------------------------------------ #


class TestSampling:
    """Parameter space sampling."""

    def test_lhs_shape(self):
        X, Y = pr.sample(
            pr.analytical.cl_flat_plate,
            bounds=[(-5, 15), (0.01, 0.1)],
            n=50,
            seed=0,
        )
        assert X.shape == (50, 2)
        assert Y.shape == (50, 1)

    def test_uniform_shape(self):
        X, Y = pr.sample(
            pr.analytical.cl_flat_plate,
            bounds=[(-5, 15), (0.01, 0.1)],
            n=50,
            method="uniform",
            seed=0,
        )
        assert X.shape == (50, 2)
        assert Y.shape == (50, 1)

    def test_bounds_respected(self):
        X, _ = pr.sample(
            pr.analytical.cl_flat_plate,
            bounds=[(-5, 15), (0.01, 0.1)],
            n=200,
            method="lhs",
            seed=0,
        )
        assert X[:, 0].min() >= -5.0
        assert X[:, 0].max() <= 15.0
        assert X[:, 1].min() >= 0.01
        assert X[:, 1].max() <= 0.1

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown sampling method"):
            pr.sample(pr.analytical.cl_flat_plate, bounds=[(0, 1)], n=10, method="bad")

    def test_sobol_shape(self):
        X, Y = pr.sample(
            pr.analytical.cl_flat_plate,
            bounds=[(-5, 15), (0.01, 0.1)],
            n=64,  # Sobol requires power-of-2 for full sequence
            method="sobol",
            seed=0,
        )
        assert X.shape == (64, 2)
        assert Y.shape == (64, 1)

    def test_sobol_bounds_respected(self):
        X, _ = pr.sample(
            pr.analytical.cl_flat_plate,
            bounds=[(-5, 15), (0.01, 0.1)],
            n=64,
            method="sobol",
            seed=0,
        )
        assert X[:, 0].min() >= -5.0
        assert X[:, 0].max() <= 15.0
        assert X[:, 1].min() >= 0.01
        assert X[:, 1].max() <= 0.1

    def test_param_count_mismatch_raises(self):
        """If bounds count != function param count, raise clear error."""

        def f(a, b, c):
            return {"y": 1.0}

        with pytest.raises(ValueError, match="3 parameters but 2 bounds"):
            pr.sample(f, bounds=[(0, 1), (0, 1)], n=5)


class TestAnalytical:
    """Analytical truth functions return correct values."""

    def test_cl_flat_plate_zero_alpha(self):
        """At zero alpha with 4% camber, CL ≈ 2π·2·0.04 ≈ 0.503."""
        result = pr.analytical.cl_flat_plate(alpha=0.0, camber=0.04)
        assert result["CL"] == pytest.approx(2 * np.pi * 0.08, rel=1e-6)

    def test_cl_flat_plate_linear(self):
        """CL scales linearly with alpha in thin airfoil theory."""
        cl1 = pr.analytical.cl_flat_plate(alpha=2.0, camber=0.0)["CL"]
        cl2 = pr.analytical.cl_flat_plate(alpha=4.0, camber=0.0)["CL"]
        assert cl2 == pytest.approx(2 * cl1, rel=1e-6)

    def test_cd_cylinder_low_re(self):
        """At very low Re, CD should be high (Stokes regime)."""
        result = pr.analytical.cd_cylinder(reynolds=0.5)
        assert result["CD"] > 10.0  # Stokes drag is large

    def test_cd_cylinder_supercritical(self):
        """At high Re, CD should be around 0.3."""
        result = pr.analytical.cd_cylinder(reynolds=1e7)
        assert result["CD"] == pytest.approx(0.3, rel=0.1)

    def test_thrust_propeller_positive(self):
        result = pr.analytical.thrust_propeller(rpm=5000, diameter=0.3, pitch=0.15)
        assert result["T"] > 0.0


class TestGPSurrogate:
    """Gaussian Process surrogate fitting and prediction."""

    def test_fit_returns_self(self, gp_surrogate):
        surrogate, _, _ = gp_surrogate
        assert surrogate._fitted

    def test_predict_shape(self, gp_surrogate):
        surrogate, _, _ = gp_surrogate
        X_test, Y_test = pr.sample(
            pr.analytical.cl_flat_plate,
            bounds=[(-5, 15), (0.01, 0.1)],
            n=20,
            seed=99,
        )
        Y_pred = surrogate.predict(X_test)
        assert Y_pred.shape == (20, 1)

    def test_r2_above_099(self, gp_surrogate):
        """GP surrogate should achieve R² > 0.99 on smooth analytical truth."""
        surrogate, _, _ = gp_surrogate
        X_test, Y_test = pr.sample(
            pr.analytical.cl_flat_plate,
            bounds=[(-5, 15), (0.01, 0.1)],
            n=30,
            seed=99,
        )
        report = surrogate.validate(X_test, Y_test)
        assert report["CL"]["r2"] > 0.99, f"GP R² = {report['CL']['r2']:.4f}"

    def test_rmse_small(self, gp_surrogate):
        """GP RMSE should be very small on smooth function."""
        surrogate, _, _ = gp_surrogate
        X_test, Y_test = pr.sample(
            pr.analytical.cl_flat_plate,
            bounds=[(-5, 15), (0.01, 0.1)],
            n=30,
            seed=99,
        )
        report = surrogate.validate(X_test, Y_test)
        assert report["CL"]["rmse"] < 0.05, f"GP RMSE = {report['CL']['rmse']:.4f}"

    def test_repr(self, gp_surrogate):
        surrogate, _, _ = gp_surrogate
        rep = repr(surrogate)
        assert "gp" in rep
        assert "fitted" in rep

    def test_predict_before_fit_raises(self):
        s = pr.Surrogate(params=["x"], outputs=["y"], method="gp")
        with pytest.raises(RuntimeError, match="not fitted"):
            s.predict(np.array([[1.0]]))

    def test_gp_kernel_matern15(self):
        """GP with Matérn 1.5 kernel should still fit well."""
        X, Y = pr.sample(
            pr.analytical.cl_flat_plate,
            bounds=[(-5, 15), (0.01, 0.1)],
            n=100,
            seed=42,
        )
        s = pr.Surrogate(params=["alpha", "camber"], outputs=["CL"], method="gp", gp_kernel="matern15")
        s.fit(X, Y, verbose=False)
        X_test, Y_test = pr.sample(pr.analytical.cl_flat_plate, bounds=[(-5,15),(0.01,0.1)], n=20, seed=99)
        report = s.validate(X_test, Y_test)
        assert report["CL"]["r2"] > 0.99

    def test_gp_kernel_matern25(self):
        """GP with Matérn 1/2 kernel (nu=0.5) — rougher but valid."""
        X, Y = pr.sample(
            pr.analytical.cl_flat_plate,
            bounds=[(-5, 15), (0.01, 0.1)],
            n=100,
            seed=42,
        )
        s = pr.Surrogate(params=["alpha", "camber"], outputs=["CL"], method="gp", gp_kernel="matern25")
        s.fit(X, Y, verbose=False)
        X_test, Y_test = pr.sample(pr.analytical.cl_flat_plate, bounds=[(-5,15),(0.01,0.1)], n=20, seed=99)
        report = s.validate(X_test, Y_test)
        assert report["CL"]["r2"] > 0.95  # rougher kernel, lower tolerance

    def test_gp_kernel_matern52(self):
        """GP with Matérn 5/2 kernel should fit well on smooth data."""
        X, Y = pr.sample(
            pr.analytical.cl_flat_plate,
            bounds=[(-5, 15), (0.01, 0.1)],
            n=100,
            seed=42,
        )
        s = pr.Surrogate(params=["alpha", "camber"], outputs=["CL"], method="gp", gp_kernel="matern52")
        s.fit(X, Y, verbose=False)
        X_test, Y_test = pr.sample(pr.analytical.cl_flat_plate, bounds=[(-5,15),(0.01,0.1)], n=20, seed=99)
        report = s.validate(X_test, Y_test)
        assert report["CL"]["r2"] > 0.99

    def test_invalid_kernel_raises(self):
        """Unknown kernel name should raise a clear error."""
        s = pr.Surrogate(params=["x"], outputs=["y"], method="gp", gp_kernel="bad_kernel")
        X = np.array([[1.0], [2.0], [3.0]])
        Y = np.array([[0.5], [2.0], [4.5]])
        with pytest.raises(ValueError, match="kernel"):
            s.fit(X, Y, verbose=False)

    def test_invalid_method_raises(self):
        """Unknown surrogate method should raise ValueError."""
        with pytest.raises(ValueError, match="method"):
            pr.Surrogate(params=["x"], outputs=["y"], method="fantasy")

    def test_shape_mismatch_raises(self):
        """X rows != Y rows should raise clear error."""
        s = pr.Surrogate(params=["a", "b"], outputs=["y"], method="gp")
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        Y = np.array([[1.0]])  # 1 row vs 2
        with pytest.raises(ValueError, match="same number|X must.*Y|samples"):
            s.fit(X, Y, verbose=False)

    def test_param_dim_mismatch_raises(self):
        """X columns != len(params) should raise clear error."""
        s = pr.Surrogate(params=["a", "b", "c"], outputs=["y"], method="gp")
        X = np.array([[1.0, 2.0]])  # 2 columns vs 3 params
        Y = np.array([[1.0]])
        with pytest.raises(ValueError, match="shape.*3.*2|X must"):
            s.fit(X, Y, verbose=False)


class TestMLPSurrogate:
    """MLP surrogate fitting and prediction."""

    def test_fit_returns_self(self, mlp_surrogate):
        surrogate, _, _ = mlp_surrogate
        assert surrogate._fitted

    def test_predict_shape(self, mlp_surrogate):
        surrogate, _, _ = mlp_surrogate
        X_test, Y_test = pr.sample(
            pr.analytical.cl_flat_plate,
            bounds=[(-5, 15), (0.01, 0.1)],
            n=20,
            seed=99,
        )
        Y_pred = surrogate.predict(X_test)
        assert Y_pred.shape == (20, 1)

    def test_r2_above_099(self, mlp_surrogate):
        """MLP surrogate should achieve R² > 0.99 on smooth analytical truth."""
        surrogate, _, _ = mlp_surrogate
        X_test, Y_test = pr.sample(
            pr.analytical.cl_flat_plate,
            bounds=[(-5, 15), (0.01, 0.1)],
            n=30,
            seed=99,
        )
        report = surrogate.validate(X_test, Y_test)
        assert report["CL"]["r2"] > 0.99, f"MLP R² = {report['CL']['r2']:.4f}"

    def test_rmse_small(self, mlp_surrogate):
        surrogate, _, _ = mlp_surrogate
        X_test, Y_test = pr.sample(
            pr.analytical.cl_flat_plate,
            bounds=[(-5, 15), (0.01, 0.1)],
            n=30,
            seed=99,
        )
        report = surrogate.validate(X_test, Y_test)
        assert report["CL"]["rmse"] < 0.05, f"MLP RMSE = {report['CL']['rmse']:.4f}"

    def test_mlp_repr(self, mlp_surrogate):
        surrogate, _, _ = mlp_surrogate
        rep = repr(surrogate)
        assert "mlp" in rep
        assert "fitted" in rep


class TestExport:
    """ONNX export."""

    def test_export_gp_raises(self, gp_surrogate, tmp_path):
        """GP models cannot be exported to ONNX (non-parametric)."""
        surrogate, _, _ = gp_surrogate
        out = tmp_path / "model.onnx"
        with pytest.raises(RuntimeError, match="ONNX export is only supported for method='mlp'"):
            surrogate.export(str(out))

    def test_export_mlp_creates_file(self, mlp_surrogate, tmp_path):
        """MLP surrogates export to valid ONNX files."""
        surrogate, _, _ = mlp_surrogate
        out = tmp_path / "model.onnx"
        surrogate.export(str(out))
        cl_file = tmp_path / "model__CL.onnx"
        assert cl_file.exists()
        assert cl_file.stat().st_size > 0

    def test_export_before_fit_raises(self, tmp_path):
        s = pr.Surrogate(params=["x"], outputs=["y"], method="gp")
        with pytest.raises(RuntimeError, match="not fitted"):
            s.export(str(tmp_path / "model.onnx"))


class TestMultipleOutputs:
    """Multi-output surrogate with analytical truth."""

    @staticmethod
    def two_output_func(x: float) -> dict[str, float]:
        """Simple analytical function with two outputs for testing."""
        return {"y1": x**2, "y2": x**3}

    def test_gp_two_outputs(self):
        X, Y = pr.sample(self.two_output_func, bounds=[(-3, 3)], n=100, seed=0)
        s = pr.Surrogate(params=["x"], outputs=["y1", "y2"], method="gp")
        s.fit(X, Y, verbose=False)

        X_test, Y_test = pr.sample(self.two_output_func, bounds=[(-3, 3)], n=30, seed=1)
        report = s.validate(X_test, Y_test)

        assert report["y1"]["r2"] > 0.99, f"y1 R²={report['y1']['r2']:.4f}"
        assert report["y2"]["r2"] > 0.99, f"y2 R²={report['y2']['r2']:.4f}"


class TestTreeSurrogate:
    """Random Forest and Gradient Boosting surrogate backends."""

    @staticmethod
    def _quadratic(x):
        """f(x) = x²"""
        return {"y": float(x**2)}

    def test_rf_surrogate(self):
        X, Y = pr.sample(self._quadratic, bounds=[(-3, 3)], n=80, seed=42)
        s = pr.Surrogate(params=["x"], outputs=["y"], method="rf")
        s.fit(X, Y)

        X_test, Y_test = pr.sample(self._quadratic, bounds=[(-3, 3)], n=30, seed=1)
        report = s.validate(X_test, Y_test)
        assert report["y"]["r2"] > 0.9, f"RF R²={report['y']['r2']:.4f}"

    def test_gb_surrogate(self):
        X, Y = pr.sample(self._quadratic, bounds=[(-3, 3)], n=80, seed=42)
        s = pr.Surrogate(params=["x"], outputs=["y"], method="gb")
        s.fit(X, Y)

        X_test, Y_test = pr.sample(self._quadratic, bounds=[(-3, 3)], n=30, seed=1)
        report = s.validate(X_test, Y_test)
        assert report["y"]["r2"] > 0.9, f"GB R²={report['y']['r2']:.4f}"

    def test_rf_uncertainty(self):
        X, Y = pr.sample(self._quadratic, bounds=[(-3, 3)], n=80, seed=42)
        s = pr.Surrogate(params=["x"], outputs=["y"], method="rf")
        s.fit(X, Y)

        X_test = np.linspace(-3, 3, 10).reshape(-1, 1)
        y_mu, y_std = s.predict_with_uncertainty(X_test)
        assert y_mu.shape == (10, 1)
        assert y_std.shape == (10, 1)
        assert np.all(y_std > 0), "uncertainty must be positive"

    def test_gb_uncertainty_raises(self):
        X, Y = pr.sample(self._quadratic, bounds=[(-3, 3)], n=80, seed=42)
        s = pr.Surrogate(params=["x"], outputs=["y"], method="gb")
        s.fit(X, Y)
        X_test = np.linspace(-3, 3, 5).reshape(-1, 1)
        with pytest.raises(NotImplementedError, match="quantile regression"):
            s.predict_with_uncertainty(X_test)

    def test_rf_multi_output(self):
        X, Y = pr.sample(TestMultipleOutputs.two_output_func, bounds=[(-3, 3)], n=100, seed=42)
        s = pr.Surrogate(params=["x"], outputs=["y1", "y2"], method="rf")
        s.fit(X, Y)
        X_test, Y_test = pr.sample(
            TestMultipleOutputs.two_output_func, bounds=[(-3, 3)], n=30, seed=1
        )
        report = s.validate(X_test, Y_test)
        assert report["y1"]["r2"] > 0.95
        assert report["y2"]["r2"] > 0.95
