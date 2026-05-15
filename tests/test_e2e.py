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


class TestExport:
    """ONNX export."""

    def test_export_gp_raises(self, gp_surrogate, tmp_path):
        """GP models cannot be exported to ONNX (non-parametric)."""
        surrogate, _, _ = gp_surrogate
        out = tmp_path / "model.onnx"
        with pytest.raises(RuntimeError, match="ONNX export is not supported for Gaussian"):
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
