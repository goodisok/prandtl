"""Multi-fidelity surrogate modelling via Co-Kriging (Kennedy & O'Hagan, 2000).

Combines cheap low-fidelity data with a few expensive high-fidelity points
using an autoregressive formulation::

    Z_H(x) = ρ · Z_L(x) + δ(x)

where Z_L is a GP fitted to low-fidelity data, ρ is a scaling factor, and
δ is a GP fitted to the discrepancy between scaled low-fidelity predictions
and the high-fidelity data.

References
----------
Kennedy, M. C., & O'Hagan, A. (2000). Predicting the output from a complex
computer code when fast approximations are available. *Biometrika*, 87(1), 1-13.
"""

from __future__ import annotations

import numpy as np
import torch

from prandtl._gaussian import _train_gp, _ExactGP


class CoKriging:
    """Multi-fidelity co-kriging surrogate model.

    Fits a Gaussian Process on low-fidelity data and another on the
    discrepancy between scaled low-fidelity predictions and high-fidelity
    observations.

    Parameters
    ----------
    params : list of str
        Names of input parameters (must match column order in X).
    output : str
        Name of the output variable.
    """

    def __init__(self, params: list[str], output: str) -> None:
        self._params = list(params)
        self._output = output
        self._d = len(params)

        self._gp_low: _ExactGP | None = None
        self._gp_delta: _ExactGP | None = None
        self._rho: float | None = None

        # Standardisation stats
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self._y_mean_low: float | None = None
        self._y_std_low: float | None = None
        self._y_mean_high: float | None = None
        self._y_std_high: float | None = None

        self._fitted = False

    @property
    def rho(self) -> float | None:
        """Estimated scaling factor ρ (only available after fitting)."""
        return self._rho

    def fit(
        self,
        X_low: np.ndarray,
        Y_low: np.ndarray,
        X_high: np.ndarray,
        Y_high: np.ndarray,
        *,
        n_iter: int = 200,
        lr: float = 0.1,
        verbose: bool = True,
    ) -> "CoKriging":
        """Fit the co-kriging model.

        Two-stage Kennedy & O'Hagan approach:
        1. Fit a GP on low-fidelity data.
        2. Optimise the scaling factor ρ and fit a discrepancy GP on
           Y_high - ρ · GP_low(X_high).

        Parameters
        ----------
        X_low : ndarray of shape (n_low, n_params)
            Low-fidelity input points.
        Y_low : ndarray of shape (n_low,)
            Low-fidelity output values.
        X_high : ndarray of shape (n_high, n_params)
            High-fidelity input points.
        Y_high : ndarray of shape (n_high,)
            High-fidelity output values.
        n_iter : int
            Training iterations for each GP.
        lr : float
            Learning rate for GP hyperparameter optimisation.
        verbose : bool
            Print fitting progress.

        Returns
        -------
        self
        """
        X_low = np.asarray(X_low, dtype=np.float64)
        Y_low = np.asarray(Y_low, dtype=np.float64).ravel()
        X_high = np.asarray(X_high, dtype=np.float64)
        Y_high = np.asarray(Y_high, dtype=np.float64).ravel()

        if X_low.ndim != 2 or X_low.shape[1] != self._d:
            raise ValueError(f"X_low must be (n, {self._d}), got {X_low.shape}")
        if X_high.ndim != 2 or X_high.shape[1] != self._d:
            raise ValueError(f"X_high must be (n, {self._d}), got {X_high.shape}")
        if len(X_low) != len(Y_low):
            raise ValueError(f"X_low and Y_low have mismatched lengths: {len(X_low)} vs {len(Y_low)}")
        if len(X_high) != len(Y_high):
            raise ValueError(f"X_high and Y_high have mismatched lengths: {len(X_high)} vs {len(Y_high)}")

        # Combined standardisation
        X_all = np.vstack([X_low, X_high])
        self._x_mean = X_all.mean(axis=0)
        self._x_std = X_all.std(axis=0, ddof=0)
        self._x_std[self._x_std < 1e-12] = 1.0

        self._y_mean_low = float(Y_low.mean())
        self._y_std_low = float(Y_low.std(ddof=0))
        if self._y_std_low < 1e-12:
            self._y_std_low = 1.0

        self._y_mean_high = float(Y_high.mean())
        self._y_std_high = float(Y_high.std(ddof=0))
        if self._y_std_high < 1e-12:
            self._y_std_high = 1.0

        # Scale inputs
        X_low_s = (X_low - self._x_mean) / self._x_std
        X_high_s = (X_high - self._x_mean) / self._x_std

        # ----- Stage 1: fit low-fidelity GP -----
        Y_low_s = (Y_low - self._y_mean_low) / self._y_std_low

        x_low_t = torch.tensor(X_low_s, dtype=torch.float32)
        y_low_t = torch.tensor(Y_low_s, dtype=torch.float32)

        if verbose:
            print("[Co-Kriging] Stage 1/2: fitting low-fidelity GP ...")

        self._gp_low = _ExactGP(x_low_t, y_low_t, kernel="rbf")
        _train_gp(self._gp_low, x_low_t, y_low_t, n_iter=n_iter, lr=lr, verbose=False)

        # Posterior mean of low-fidelity GP at high-fidelity points
        self._gp_low.eval()
        with torch.no_grad():
            x_high_t = torch.tensor(X_high_s, dtype=torch.float32)
            mu_low_at_high_s = self._gp_low(x_high_t).mean.numpy()

        # ----- Stage 2: optimise ρ and fit discrepancy GP -----
        if verbose:
            print("[Co-Kriging] Stage 2/2: fitting discrepancy GP ...")

        # Simple approach: estimate ρ by linear regression Y_H ~ μ_L(X_H)
        # ρ = cov(Y_H, μ_L) / var(μ_L)
        mu_low_at_high = mu_low_at_high_s * self._y_std_low + self._y_mean_low
        rho_candidates = np.linspace(0.1, 3.0, 30)
        best_loss = np.inf
        best_rho = 1.0

        for rho in rho_candidates:
            discrepancy = Y_high - rho * mu_low_at_high
            # Score: variance of discrepancy (lower is better — ρ explains more)
            loss = float(discrepancy.var())
            if loss < best_loss:
                best_loss = loss
                best_rho = rho

        self._rho = best_rho

        # Scale discrepancy
        discrepancy = Y_high - self._rho * mu_low_at_high
        discrepancy_mean = float(discrepancy.mean())
        discrepancy_std = float(discrepancy.std(ddof=0))
        if discrepancy_std < 1e-12:
            discrepancy_std = 1.0

        discrepancy_s = (discrepancy - discrepancy_mean) / discrepancy_std

        d_t = torch.tensor(discrepancy_s, dtype=torch.float32)
        self._gp_delta = _ExactGP(x_high_t, d_t, kernel="rbf")
        _train_gp(self._gp_delta, x_high_t, d_t, n_iter=n_iter, lr=lr, verbose=False)

        # Store discrepancy un-standardisation stats
        self._delta_mean = discrepancy_mean
        self._delta_std = discrepancy_std

        self._fitted = True
        return self

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Predict the high-fidelity output at given points.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_params)
            Input points.
        return_std : bool
            If True, also return the predictive standard deviation.

        Returns
        -------
        Y_mean : ndarray of shape (n_samples,)
            Predictive mean of the high-fidelity output.
        Y_std : ndarray of shape (n_samples,) or None
            Predictive standard deviation (only if ``return_std=True``).
        """
        self._check_fitted()

        X = np.asarray(X, dtype=np.float64)
        X_s = (X - self._x_mean) / self._x_std
        x_t = torch.tensor(X_s, dtype=torch.float32)

        self._gp_low.eval()
        self._gp_delta.eval()

        with torch.no_grad():
            dist_low = self._gp_low(x_t)
            dist_delta = self._gp_delta(x_t)

            mu_low_s = dist_low.mean.numpy()
            mu_delta_s = dist_delta.mean.numpy()

        # Un-standardise low-fidelity
        mu_low = mu_low_s * self._y_std_low + self._y_mean_low
        mu_delta = mu_delta_s * self._delta_std + self._delta_mean

        Y_mean = self._rho * mu_low + mu_delta

        if return_std:
            sigma_low_s = dist_low.stddev.numpy()
            sigma_delta_s = dist_delta.stddev.numpy()

            sigma_low = sigma_low_s * self._y_std_low
            sigma_delta = sigma_delta_s * self._delta_std

            # Independent GPs: variance adds
            Y_std = np.sqrt((self._rho * sigma_low) ** 2 + sigma_delta**2)
            return Y_mean.ravel(), Y_std.ravel()

        return Y_mean.ravel(), None

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "CoKriging model has not been fitted yet. Call .fit() first."
            )

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        rho_str = f", ρ={self._rho:.3f}" if self._rho else ""
        return (
            f"CoKriging(params={self._params!r}, output={self._output!r}"
            f"{rho_str}, {status})"
        )
