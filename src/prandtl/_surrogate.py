"""Gaussian Process surrogate model backend powered by GPyTorch.

Provides a GP regression model with automatic kernel selection and
training loop. Designed to be used internally by the ``Surrogate`` class.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ._gaussian import _ExactGP, _train_gp
from ._neural import _MLP, _train_mlp


class Surrogate:
    """CFD surrogate model with a scikit-learn-like interface.

    Wraps Gaussian Process (GPyTorch) and neural network (PyTorch) backends
    behind a unified API.

    Parameters
    ----------
    params : list of str
        Names of input parameters, e.g. ``['alpha', 'mach', 'camber']``.
    outputs : list of str
        Names of output quantities, e.g. ``['CL', 'CD']``.
    method : str
        Backend model type: ``'gp'`` (Gaussian Process, default),
        ``'mlp'`` (multi-layer perceptron), ``'rf'`` (Random Forest),
        or ``'gb'`` (Gradient Boosting).
    gp_kernel : str or None
        GP kernel type: ``'rbf'`` (default), ``'matern15'``, ``'matern25'``,
        ``'matern52'``. Only used when ``method='gp'``.
    device : str or torch.device
        Device to run on: ``'cpu'`` (default) or ``'cuda'`` / ``'cuda:0'``.
        PyTorch CUDA backend must be available for GPU acceleration. Set to
        ``'cuda'`` when training on large datasets (1000+ points) with
        ``method='mlp'`` for significant speedup.
    """

    def __init__(
        self,
        params: list[str],
        outputs: list[str],
        method: str = "gp",
        gp_kernel: str | None = "rbf",
        device: str = "cpu",
    ) -> None:
        if method not in ("gp", "mlp", "rf", "gb"):
            raise ValueError(f"Unknown method: {method!r}. Use 'gp', 'mlp', 'rf', or 'gb'.")

        self._params = list(params)
        self._outputs = list(outputs)
        self._method = method
        self._gp_kernel = gp_kernel
        self._device = torch.device(device)
        self._models: dict[str, Any] = {}  # one model per output
        self._x_mean: np.ndarray | None = None
        self._x_std: np.ndarray | None = None
        self._y_mean: dict[str, float] = {}
        self._y_std: dict[str, float] = {}
        self._fitted = False

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        n_iter: int = 200,
        lr: float = 0.1,
        verbose: bool = False,
        physics: list | None = None,
    ) -> Surrogate:
        """Train the surrogate model on (X, Y) data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_params)
            Input design matrix.
        Y : ndarray of shape (n_samples, n_outputs)
            Output target matrix — column order must match ``self._outputs``.
        n_iter : int
            Number of training iterations (GP marginal likelihood optimization
            steps, or MLP epochs).
        lr : float
            Learning rate for optimizer.
        verbose : bool
            If True, print per-iteration loss during training.
        physics : list of _PhysicsConstraint or None
            Physics-informed penalty constraints. Only supported for
            ``method='mlp'``. Each constraint computes a penalty term from
            (X_scaled, Y_pred_scaled) and adds it to the training loss.

            Supported constraints: ``Monotonicity``, ``Convexity``,
            ``BoundaryValue``, ``CustomConstraint``.

        Returns
        -------
        Surrogate
            Self, for method chaining.

        Raises
        ------
        ValueError
            If ``physics`` constraints are provided with ``method='gp'`` —
            GP uses marginal likelihood optimization, not loss-term penalties.
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        if X.ndim != 2 or X.shape[1] != len(self._params):
            raise ValueError(f"X must have shape (n, {len(self._params)}), got {X.shape}")
        if Y.ndim != 2 or Y.shape[1] != len(self._outputs):
            raise ValueError(f"Y must have shape (n, {len(self._outputs)}), got {Y.shape}")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"X and Y must have the same number of samples, "
                f"got X: {X.shape[0]}, Y: {Y.shape[0]}"
            )

        # Physics constraints are only supported for MLP
        if physics is not None and self._method in ("gp", "rf", "gb"):
            raise ValueError(
                f"Physics constraints are only supported with method='mlp', "
                f"not '{self._method}'."
            )

        # Scale BoundaryValue constraints to match training data
        from ._physics import BoundaryValue

        physics_constraints = []
        if physics is not None:
            for c in physics:
                if isinstance(c, BoundaryValue):
                    # Scale boundary points and values consistently with training data
                    bdy_pts = np.asarray(c._raw_points, dtype=np.float64)
                    bdy_vals = np.asarray(c._raw_values, dtype=np.float64)
                    # Will be re-scaled per-output below; store raw for now
                    physics_constraints.append(
                        {"constraint": c, "raw_points": bdy_pts, "raw_values": bdy_vals}
                    )
                else:
                    physics_constraints.append({"constraint": c})
        else:
            physics_constraints = []

        # Standardize inputs
        self._x_mean = X.mean(axis=0)
        self._x_std = X.std(axis=0, ddof=1)
        self._x_std[self._x_std == 0] = 1.0  # guard against constant params
        X_scaled = (X - self._x_mean) / self._x_std

        # Train one model per output
        for j, name in enumerate(self._outputs):
            y_col = Y[:, j].copy()
            self._y_mean[name] = float(y_col.mean())
            self._y_std[name] = float(y_col.std(ddof=1))
            if self._y_std[name] < 1e-12:
                self._y_std[name] = 1.0

            if self._method in ("rf", "gb"):
                # Tree models: direct fit on raw data (no scaling needed)
                y_raw = y_col
                if self._method == "rf":
                    from sklearn.ensemble import RandomForestRegressor
                    model = RandomForestRegressor(
                        n_estimators=200, random_state=42, n_jobs=-1,
                    )
                else:
                    from sklearn.ensemble import GradientBoostingRegressor
                    model = GradientBoostingRegressor(
                        n_estimators=200, max_depth=5, learning_rate=0.05,
                        random_state=42,
                    )
                model.fit(X, y_raw)
                self._models[name] = model
                continue

            # GP/MLP: scale and use PyTorch
            y_scaled = (y_col - self._y_mean[name]) / self._y_std[name]
            x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self._device)
            y_tensor = torch.tensor(y_scaled, dtype=torch.float32).to(self._device)

            if self._method == "gp":
                model = _ExactGP(x_tensor, y_tensor, kernel=self._gp_kernel or "rbf")
                model = model.to(self._device)
                _train_gp(model, x_tensor, y_tensor, n_iter=n_iter, lr=lr, verbose=verbose)
            else:  # mlp
                model = _MLP(in_dim=len(self._params)).to(self._device)

                # Build per-output physics constraints with scaled boundary values
                scaled_constraints = []
                for entry in physics_constraints:
                    c = entry["constraint"]
                    if isinstance(c, BoundaryValue):
                        bdy_scaled = (entry["raw_points"] - self._x_mean) / self._x_std
                        bdy_val_scaled = (entry["raw_values"] - self._y_mean[name]) / self._y_std[name]
                        c.points = torch.tensor(bdy_scaled, dtype=torch.float32).to(self._device)
                        c.values = torch.tensor(bdy_val_scaled, dtype=torch.float32).to(self._device)
                        scaled_constraints.append(c)
                    else:
                        scaled_constraints.append(c)

                _train_mlp(
                    model, x_tensor, y_tensor,
                    n_iter=n_iter, lr=lr, verbose=verbose,
                    physics_constraints=scaled_constraints if scaled_constraints else None,
                )

            self._models[name] = model

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return surrogate predictions for given inputs.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_params)
            Input points to predict at.

        Returns
        -------
        Y_pred : ndarray of shape (n_samples, n_outputs)
            Predicted outputs in the same column order as ``self._outputs``.
        """
        self._check_fitted()

        X = np.asarray(X, dtype=np.float64)
        X_scaled = (X - self._x_mean) / self._x_std
        x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self._device)

        preds = []
        for name in self._outputs:
            model = self._models[name]

            if self._method in ("rf", "gb"):
                # Tree models: direct prediction on raw data
                y_pred = model.predict(X)
                preds.append(y_pred.ravel())
                continue

            model.eval()
            x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self._device)
            with torch.no_grad():
                if self._method == "gp":
                    y_pred_scaled = model(x_tensor).mean.cpu().numpy()
                else:
                    y_pred_scaled = model(x_tensor).cpu().numpy()
            # Un-standardize
            y_pred = y_pred_scaled * self._y_std[name] + self._y_mean[name]
            preds.append(y_pred.ravel())

        return np.column_stack(preds)

    def predict_with_uncertainty(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return predictive mean and standard deviation.

        For Gaussian Process (``method='gp'``), the predictive distribution is
        a full Gaussian — the standard deviation captures model uncertainty
        (epistemic uncertainty), growing in regions far from training data.

        For MLP (``method='mlp'``), this is not yet supported because a single
        deterministic forward pass has no built-in uncertainty estimate.
        Use ``method='gp'`` or ensemble-based approaches for uncertainty.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_params)
            Input points to predict at.

        Returns
        -------
        Y_mean : ndarray of shape (n_samples, n_outputs)
            Predictive mean — same as ``predict()``.
        Y_std : ndarray of shape (n_samples, n_outputs)
            Predictive standard deviation (one sigma). Values are in the
            original output scale (un-standardized).

        Raises
        ------
        NotImplementedError
            If ``method='mlp'`` — use GP for uncertainty quantification.
        """
        self._check_fitted()

        if self._method == "mlp":
            raise NotImplementedError(
                "Uncertainty quantification is only available for method='gp' or 'rf'. "
                "Use method='gp' for principled uncertainty quantification "
                "or method='rf' for tree-variance estimates."
            )

        X = np.asarray(X, dtype=np.float64)

        if self._method == "rf":
            means, stds = [], []
            for name in self._outputs:
                model = self._models[name]
                tree_preds = np.array([tree.predict(X) for tree in model.estimators_])
                means.append(tree_preds.mean(axis=0))
                stds.append(tree_preds.std(axis=0, ddof=1))
            return np.column_stack(means), np.column_stack(stds)

        if self._method == "gb":
            raise NotImplementedError(
                "Uncertainty for method='gb' requires quantile regression. "
                "Use prandtl.GradientBoosting(...).fit_with_uncertainty(X, Y) "
                "for direct quantile-based uncertainty."
            )

        # GP: analytic posterior uncertainty
        X_scaled = (X - self._x_mean) / self._x_std
        x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self._device)

        means = []
        stds = []
        for name in self._outputs:
            model = self._models[name]
            model.eval()
            with torch.no_grad():
                dist = model(x_tensor)
                y_mean_scaled = dist.mean.cpu().numpy()
                y_std_scaled = dist.stddev.cpu().numpy()
            # Un-standardize: mean scaled back, std only needs y_std multiplier
            y_mean = y_mean_scaled * self._y_std[name] + self._y_mean[name]
            y_std = y_std_scaled * self._y_std[name]
            means.append(y_mean.ravel())
            stds.append(y_std.ravel())

        return np.column_stack(means), np.column_stack(stds)

    def validate(self, X_test: np.ndarray, Y_test: np.ndarray) -> dict[str, dict[str, float]]:
        """Compute validation metrics on held-out test data.

        Parameters
        ----------
        X_test : ndarray of shape (n_test, n_params)
            Test input points.
        Y_test : ndarray of shape (n_test, n_outputs)
            True output values for test points.

        Returns
        -------
        report : dict
            Nested dict: ``{output_name: {"r2": float, "rmse": float, "max_error": float}}``.
        """
        self._check_fitted()

        Y_pred = self.predict(X_test)
        Y_test = np.asarray(Y_test, dtype=np.float64)

        report = {}
        for j, name in enumerate(self._outputs):
            y_true = Y_test[:, j]
            y_pred = Y_pred[:, j]

            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else float("nan")

            report[name] = {
                "r2": float(r2),
                "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
                "max_error": float(np.max(np.abs(y_true - y_pred))),
            }

        return report

    # ------------------------------------------------------------------ #
    #  Export
    # ------------------------------------------------------------------ #

    def export(self, path: str) -> None:
        """Export surrogate models to ONNX format.

        One ONNX file per output: ``path_stem__CL.onnx``, ``path_stem__CD.onnx``, etc.

        .. note::

            Only supported for ``method='mlp'``. Gaussian Process models are
            non-parametric — they require the full training dataset for inference
            and cannot be exported to ONNX.

        Parameters
        ----------
        path : str
            Base output path. Output names are inserted before the extension.
        """
        self._check_fitted()

        if self._method != "mlp":
            raise RuntimeError(
                f"ONNX export is only supported for method='mlp', not '{self._method}'. "
                f"GP models are non-parametric and tree models (rf/gb) are scikit-learn based."
            )

        try:
            import onnx  # noqa: F401
            import onnxruntime  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "onnx and onnxruntime are required for export. "
                "Install with: pip install prandtl[export]"
            ) from exc

        import os

        base, ext = os.path.splitext(path)
        if not ext:
            ext = ".onnx"

        x_sample = torch.randn(1, len(self._params), dtype=torch.float32).to(self._device)

        for name in self._outputs:
            model = self._models[name]
            model.eval()

            out_path = f"{base}__{name}{ext}"

            torch.onnx.export(
                model,
                x_sample,
                out_path,
                input_names=["X"],
                output_names=[name],
                dynamic_axes={"X": {0: "batch"}, name: {0: "batch"}},
                opset_version=15,
            )

    # ------------------------------------------------------------------ #
    #  Internals
    # ------------------------------------------------------------------ #

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Surrogate not fitted. Call .fit() first.")

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return (
            f"Surrogate(params={self._params!r}, outputs={self._outputs!r}, "
            f"method={self._method!r}, device={str(self._device)!r}, {status})"
        )
