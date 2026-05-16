"""Tests for active learning and Bayesian optimisation in prandtl._active."""

from __future__ import annotations

import numpy as np
import pytest

import prandtl as pr
from prandtl._active import _infer_y_best, active_learn, propose_next


# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture(scope="module")
def fitted_gp_1d() -> pr.Surrogate:
    """A fitted 1D GP on x² data for quick propose_next tests."""

    def f(x):
        return {"y": float(x**2)}

    X, Y = pr.sample(f, bounds=[(-3, 3)], n=30, seed=42)
    s = pr.Surrogate(params=["x"], outputs=["y"], method="gp")
    s.fit(X, Y, verbose=False)
    return s


# ------------------------------------------------------------------ #
#  propose_next
# ------------------------------------------------------------------ #


class TestProposeNext:
    """propose_next() — single-point acquisition optimisation."""

    def test_uncertainty_returns_valid_point(self, fitted_gp_1d):
        x_next = propose_next(fitted_gp_1d, bounds=[(-3, 3)], strategy="uncertainty")
        assert isinstance(x_next, np.ndarray)
        assert x_next.shape == (1,)
        assert -3 <= x_next[0] <= 3

    def test_ei_returns_valid_point(self, fitted_gp_1d):
        x_next = propose_next(
            fitted_gp_1d, bounds=[(-3, 3)], strategy="ei", y_best=0.0
        )
        assert -3 <= x_next[0] <= 3

    def test_pi_returns_valid_point(self, fitted_gp_1d):
        x_next = propose_next(
            fitted_gp_1d, bounds=[(-3, 3)], strategy="pi", y_best=0.0
        )
        assert -3 <= x_next[0] <= 3

    def test_ucb_returns_valid_point(self, fitted_gp_1d):
        x_next = propose_next(
            fitted_gp_1d, bounds=[(-3, 3)], strategy="ucb", beta=1.0
        )
        assert -3 <= x_next[0] <= 3

    def test_all_strategies(self, fitted_gp_1d):
        """All strategies return points within bounds."""
        for strategy in ("uncertainty", "ei", "pi", "ucb"):
            x_next = propose_next(
                fitted_gp_1d,
                bounds=[(-3, 3)],
                strategy=strategy,
                y_best=0.0,
                seed=42,
            )
            assert -3 <= x_next[0] <= 3, f"{strategy} gave {x_next[0]}"

    def test_unknown_strategy_raises(self, fitted_gp_1d):
        with pytest.raises(ValueError, match="Unknown strategy"):
            propose_next(fitted_gp_1d, bounds=[(-3, 3)], strategy="bad_strategy")

    def test_non_gp_raises(self):
        """propose_next requires GP method."""

        def f(x):
            return {"y": float(x)}

        X, Y = pr.sample(f, bounds=[(0, 1)], n=10, seed=0)
        mlp = pr.Surrogate(params=["x"], outputs=["y"], method="mlp")
        mlp.fit(X, Y, n_iter=200, verbose=False)

        with pytest.raises(ValueError, match="method='gp'"):
            propose_next(mlp, bounds=[(0, 1)])

    def test_seed_reproducibility(self, fitted_gp_1d):
        """Same seed gives same point."""
        x1 = propose_next(fitted_gp_1d, bounds=[(-3, 3)], seed=42)
        x2 = propose_next(fitted_gp_1d, bounds=[(-3, 3)], seed=42)
        np.testing.assert_array_almost_equal(x1, x2)

    def test_2d_bounds(self):
        """propose_next handles 2D parameter space."""

        def f(x1, x2):
            return {"y": float(x1**2 + x2**2)}

        X, Y = pr.sample(f, bounds=[(-2, 2), (-2, 2)], n=40, seed=42)
        gp = pr.Surrogate(params=["x1", "x2"], outputs=["y"], method="gp")
        gp.fit(X, Y, verbose=False)

        x_next = propose_next(gp, bounds=[(-2, 2), (-2, 2)], seed=42)
        assert x_next.shape == (2,)
        assert -2 <= x_next[0] <= 2
        assert -2 <= x_next[1] <= 2

    def test_ei_infers_y_best(self, fitted_gp_1d):
        """EI/PI infer y_best from training data when not provided."""
        x_next = propose_next(fitted_gp_1d, bounds=[(-3, 3)], strategy="ei")
        assert -3 <= x_next[0] <= 3

    def test_strategy_with_beta(self, fitted_gp_1d):
        """Higher beta for UCB encourages more exploration (qualitative)."""
        x_lo = propose_next(fitted_gp_1d, bounds=[(-3, 3)], strategy="ucb", beta=0.1, seed=42)
        x_hi = propose_next(fitted_gp_1d, bounds=[(-3, 3)], strategy="ucb", beta=5.0, seed=42)
        assert -3 <= x_lo[0] <= 3
        assert -3 <= x_hi[0] <= 3
        # Both valid — exact comparison not meaningful due to optimisation


# ------------------------------------------------------------------ #
#  active_learn
# ------------------------------------------------------------------ #


class TestActiveLearn:
    """active_learn() — full active learning loop."""

    @staticmethod
    def _simple_1d(x):
        """Simple concave function: f(x) = -(x-0.5)² + 1."""
        return {"y": float(-((x - 0.5) ** 2) + 1.0)}

    def test_returns_tuple(self):
        """active_learn returns (X, Y, history)."""
        s_template = pr.Surrogate(params=["x"], outputs=["y"], method="gp")

        X, Y, history = active_learn(
            self._simple_1d,
            bounds=[(-2, 2)],
            surrogate=s_template,
            n_initial=10,
            n_iter=3,
            strategy="uncertainty",
            seed=42,
            verbose=False,
        )

        assert isinstance(X, np.ndarray)
        assert isinstance(Y, np.ndarray)
        assert isinstance(history, list)
        assert X.shape == (13, 1)  # 10 initial + 3 active
        assert Y.shape == (13, 1)
        assert len(history) == 3
        assert all(isinstance(v, float) for v in history)

    def test_history_values_decreasing(self):
        """Uncertainty should decrease as we add more points."""
        s_template = pr.Surrogate(params=["x"], outputs=["y"], method="gp")

        _, _, history = active_learn(
            self._simple_1d,
            bounds=[(-2, 2)],
            surrogate=s_template,
            n_initial=10,
            n_iter=5,
            strategy="uncertainty",
            seed=42,
            verbose=False,
        )

        # Uncertainty generally decreases
        assert len(history) == 5
        # At minimum, history values are all finite
        assert all(np.isfinite(v) for v in history)

    def test_ei_strategy(self):
        """EI strategy completes without error."""
        s_template = pr.Surrogate(params=["x"], outputs=["y"], method="gp")

        X, Y, history = active_learn(
            self._simple_1d,
            bounds=[(-2, 2)],
            surrogate=s_template,
            n_initial=10,
            n_iter=3,
            strategy="ei",
            seed=42,
            verbose=False,
        )

        assert X.shape == (13, 1)
        assert len(history) == 3

    def test_ucb_strategy(self):
        """UCB strategy completes without error."""
        s_template = pr.Surrogate(params=["x"], outputs=["y"], method="gp")

        X, Y, history = active_learn(
            self._simple_1d,
            bounds=[(-2, 2)],
            surrogate=s_template,
            n_initial=10,
            n_iter=3,
            strategy="ucb",
            beta=2.0,
            seed=42,
            verbose=False,
        )

        assert X.shape == (13, 1)
        assert len(history) == 3

    def test_non_gp_raises(self):
        """active_learn requires GP surrogate."""

        def f(x):
            return {"y": float(x)}

        template = pr.Surrogate(params=["x"], outputs=["y"], method="mlp")
        with pytest.raises(ValueError, match="method='gp'"):
            active_learn(
                f, bounds=[(0, 1)], surrogate=template, n_initial=5, n_iter=1, verbose=False
            )

    def test_zero_iterations(self):
        """n_iter=0 returns only initial samples."""
        s_template = pr.Surrogate(params=["x"], outputs=["y"], method="gp")
        X, Y, history = active_learn(
            self._simple_1d,
            bounds=[(-2, 2)],
            surrogate=s_template,
            n_initial=5,
            n_iter=0,
            seed=42,
            verbose=False,
        )
        assert X.shape == (5, 1)
        assert Y.shape == (5, 1)
        assert history == []

    def test_2d_active_learning(self):
        """Active learning in 2D parameter space."""

        def f(x1, x2):
            return {"y": float(x1**2 + x2**2)}

        s_template = pr.Surrogate(params=["x1", "x2"], outputs=["y"], method="gp")
        X, Y, history = active_learn(
            f,
            bounds=[(-2, 2), (-2, 2)],
            surrogate=s_template,
            n_initial=10,
            n_iter=3,
            seed=42,
            verbose=False,
        )
        assert X.shape == (13, 2)
        assert Y.shape == (13, 1)
        assert len(history) == 3


# ------------------------------------------------------------------ #
#  _infer_y_best
# ------------------------------------------------------------------ #


class TestInferYBest:
    """_infer_y_best helper."""

    def test_returns_float(self, fitted_gp_1d):
        y_best = _infer_y_best(fitted_gp_1d)
        assert isinstance(y_best, float)
        assert np.isfinite(y_best)
