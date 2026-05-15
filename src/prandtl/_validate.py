"""Validation utilities — cross-validation, metrics, residual analysis, learning curves.

Provides domain-independent validation tools that work with any ``Surrogate``
instance. All functions are pure numpy/scipy — no additional dependencies.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as sp_stats

# ═══════════════════════════════════════════════════════════════════
#  Extended metrics
# ═══════════════════════════════════════════════════════════════════


def metrics(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict[str, float]:
    """Compute a comprehensive set of regression metrics.

    Parameters
    ----------
    Y_true : ndarray of shape (n_samples,)
        Ground-truth target values.
    Y_pred : ndarray of shape (n_samples,)
        Predicted target values.

    Returns
    -------
    report : dict
        Keys: ``r2``, ``rmse``, ``mae``, ``mape``, ``max_error``,
        ``max_relative_error``, ``explained_variance``.
    """
    Y_true = np.asarray(Y_true, dtype=np.float64).ravel()
    Y_pred = np.asarray(Y_pred, dtype=np.float64).ravel()

    residuals = Y_true - Y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((Y_true - Y_true.mean()) ** 2)

    # R² — coefficient of determination
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else float("nan")

    # RMSE
    rmse = float(np.sqrt(np.mean(residuals**2)))

    # MAE
    mae = float(np.mean(np.abs(residuals)))

    # MAPE — mean absolute percentage error (skip zeros to avoid inf)
    nonzero_mask = np.abs(Y_true) > 1e-15
    if nonzero_mask.any():
        mape = float(np.mean(np.abs(residuals[nonzero_mask] / Y_true[nonzero_mask])) * 100)
    else:
        mape = float("nan")

    # Max absolute error
    max_error = float(np.max(np.abs(residuals)))

    # Max relative error
    if nonzero_mask.any():
        rel_errors = np.abs(residuals[nonzero_mask] / Y_true[nonzero_mask])
        max_relative_error = float(np.max(rel_errors))
    else:
        max_relative_error = float("nan")

    # Explained variance
    var_true = Y_true.var(ddof=0)
    if var_true > 1e-15:
        explained_variance = 1.0 - residuals.var(ddof=0) / var_true
    else:
        explained_variance = float("nan")

    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "max_error": max_error,
        "max_relative_error": max_relative_error,
        "explained_variance": explained_variance,
    }


# ═══════════════════════════════════════════════════════════════════
#  Residual analysis
# ═══════════════════════════════════════════════════════════════════


def residual_analysis(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
) -> dict[str, Any]:
    """Statistical analysis of prediction residuals.

    Parameters
    ----------
    Y_true : ndarray of shape (n_samples,)
        Ground-truth values.
    Y_pred : ndarray of shape (n_samples,)
        Predicted values.

    Returns
    -------
    report : dict
        Keys: ``residuals`` (the raw residual array), ``mean``, ``std``,
        ``skewness``, ``kurtosis``, ``shapiro_statistic``, ``shapiro_pvalue``,
        ``max_abs_residual``, ``max_abs_residual_index``.
    """
    Y_true = np.asarray(Y_true, dtype=np.float64).ravel()
    Y_pred = np.asarray(Y_pred, dtype=np.float64).ravel()
    residuals = Y_true - Y_pred

    # Normality test (Shapiro-Wilk). Only valid for 3 ≤ n ≤ 5000.
    shapiro_stat = float("nan")
    shapiro_p = float("nan")
    n = len(residuals)
    if 3 <= n <= 5000:
        shapiro_stat, shapiro_p = sp_stats.shapiro(residuals)
        shapiro_stat = float(shapiro_stat)
        shapiro_p = float(shapiro_p)

    idx_max = int(np.argmax(np.abs(residuals)))

    return {
        "residuals": residuals,
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals, ddof=1)),
        "skewness": float(sp_stats.skew(residuals)),
        "kurtosis": float(sp_stats.kurtosis(residuals)),
        "shapiro_statistic": shapiro_stat,
        "shapiro_pvalue": shapiro_p,
        "max_abs_residual": float(np.max(np.abs(residuals))),
        "max_abs_residual_index": idx_max,
    }


# ═══════════════════════════════════════════════════════════════════
#  K-fold cross-validation
# ═══════════════════════════════════════════════════════════════════


def cross_validate(
    surrogate,  # Surrogate — avoid import to keep lazy
    X: np.ndarray,
    Y: np.ndarray,
    *,
    cv: int = 5,
    shuffle: bool = True,
    random_state: int | None = 42,
    verbose: bool = False,
) -> dict[str, Any]:
    """K-fold cross-validation on a surrogate model configuration.

    Clones the surrogate's configuration (params, outputs, method, kernel)
    and fits a fresh model on each fold. Returns aggregated metrics across
    all folds for every output.

    Parameters
    ----------
    surrogate : Surrogate
        An *unfitted* surrogate instance whose configuration is used as a
        template. The original instance is not modified.
    X : ndarray of shape (n_samples, n_params)
        Input design matrix.
    Y : ndarray of shape (n_samples, n_outputs)
        Output target matrix.
    cv : int
        Number of folds (default 5).
    shuffle : bool
        Whether to shuffle the data before splitting (default True).
    random_state : int or None
        Seed for the random shuffle.
    verbose : bool
        If True, print per-fold progress.

    Returns
    -------
    report : dict
        Keys per output, each containing: ``r2_mean``, ``r2_std``,
        ``rmse_mean``, ``rmse_std``, ``mae_mean``, ``mae_std``,
        ``max_error_mean``, ``max_error_std``, and ``fold_results``
        (list of per-fold metric dicts).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2D, got shape {Y.shape}")
    if len(X) != len(Y):
        raise ValueError(f"X and Y must have the same number of samples, got {len(X)} and {len(Y)}")
    if len(X) < cv:
        raise ValueError(f"Number of samples ({len(X)}) must be >= cv ({cv})")

    n_samples = len(X)

    # Build fold indices
    rng = np.random.RandomState(random_state)
    indices = np.arange(n_samples)
    if shuffle:
        rng.shuffle(indices)
    fold_sizes = np.full(cv, n_samples // cv, dtype=int)
    fold_sizes[: n_samples % cv] += 1

    # Extract surrogate config (clone-friendly attributes)
    params = list(surrogate._params)
    outputs = list(surrogate._outputs)
    method = surrogate._method
    gp_kernel = surrogate._gp_kernel

    # Import Surrogate lazily to avoid circular imports
    from ._surrogate import Surrogate

    fold_results: list[dict[str, dict[str, float]]] = []
    fold_maes: dict[str, list[float]] = {name: [] for name in outputs}

    current = 0
    for fold in range(cv):
        fold_size = fold_sizes[fold]
        test_idx = indices[current : current + fold_size]
        train_idx = np.concatenate([indices[:current], indices[current + fold_size :]])
        current += fold_size

        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]

        # Train a fresh clone on this fold
        fold_surr = Surrogate(params=params, outputs=outputs, method=method, gp_kernel=gp_kernel)
        fold_surr.fit(X_train, Y_train, verbose=verbose)

        # Validate
        fold_report = fold_surr.validate(X_test, Y_test)
        fold_results.append(fold_report)

        # Compute per-output MAE directly
        Y_pred_fold = fold_surr.predict(X_test)
        for j, name in enumerate(outputs):
            mae_fold = float(np.mean(np.abs(Y_test[:, j] - Y_pred_fold[:, j])))
            fold_maes[name].append(mae_fold)

        if verbose:
            r2_str = ", ".join(f"{name}={fold_report[name]['r2']:.4f}" for name in outputs)
            print(f"Fold {fold + 1}/{cv}: {r2_str}")

    # Aggregate per output
    report: dict[str, Any] = {}
    for j, name in enumerate(outputs):
        r2s = [fr[name]["r2"] for fr in fold_results]
        rmses = [fr[name]["rmse"] for fr in fold_results]
        maes = fold_maes[name]
        max_errs = [fr[name]["max_error"] for fr in fold_results]

        report[name] = {
            "r2_mean": float(np.mean(r2s)),
            "r2_std": float(np.std(r2s, ddof=1)),
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses, ddof=1)),
            "mae_mean": float(np.mean(maes)),
            "mae_std": float(np.std(maes, ddof=1)),
            "max_error_mean": float(np.mean(max_errs)),
            "max_error_std": float(np.std(max_errs, ddof=1)),
            "fold_results": fold_results,
        }

    return report


# ═══════════════════════════════════════════════════════════════════
#  Learning curve
# ═══════════════════════════════════════════════════════════════════


def learning_curve(
    surrogate,  # Surrogate
    X: np.ndarray,
    Y: np.ndarray,
    *,
    train_sizes: list[int] | None = None,
    cv: int = 5,
    shuffle: bool = True,
    random_state: int | None = 42,
    verbose: bool = False,
) -> dict[str, Any]:
    """Compute learning curve: model performance vs. training set size.

    For each training size, performs k-fold cross-validation and records
    the test-set R² and RMSE. Returns mean and standard deviation across
    folds for each size.

    Parameters
    ----------
    surrogate : Surrogate
        An *unfitted* surrogate instance used as a configuration template.
    X : ndarray of shape (n_samples, n_params)
        Input design matrix.
    Y : ndarray of shape (n_samples, n_outputs)
        Output target matrix.
    train_sizes : list of int, optional
        Training set sizes to evaluate. Default: 6 evenly-spaced sizes
        from ~10% to 100% of the data.
    cv : int
        Number of cross-validation folds per size.
    shuffle : bool
        Whether to shuffle before splitting.
    random_state : int or None
        Random seed.
    verbose : bool
        If True, print progress per training size.

    Returns
    -------
    result : dict
        Keys: ``train_sizes`` (list of int), and per-output dicts with
        ``r2_mean``, ``r2_std``, ``rmse_mean``, ``rmse_std`` arrays of
        shape (n_sizes,).
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n_samples = len(X)

    if train_sizes is None:
        # 6 sizes from ~10% up to all data
        fracs = np.linspace(0.1, 1.0, 6)
        train_sizes = [max(cv + 1, int(n_samples * f)) for f in fracs]
    train_sizes = [s for s in train_sizes if s <= n_samples]

    params = list(surrogate._params)
    outputs = list(surrogate._outputs)
    method = surrogate._method
    gp_kernel = surrogate._gp_kernel

    from ._surrogate import Surrogate

    # Accumulate per-size stats
    r2_per_size: dict[str, list[float]] = {name: [] for name in outputs}
    r2_std_per_size: dict[str, list[float]] = {name: [] for name in outputs}
    rmse_per_size: dict[str, list[float]] = {name: [] for name in outputs}
    rmse_std_per_size: dict[str, list[float]] = {name: [] for name in outputs}

    for size in train_sizes:
        # Subsample data for this iteration
        idx = np.arange(n_samples)
        if shuffle:
            np.random.RandomState(random_state).shuffle(idx)
        idx = idx[:size]
        X_sub = X[idx]
        Y_sub = Y[idx]

        # Cross-validate at this size
        cv_report = cross_validate(
            Surrogate(params=params, outputs=outputs, method=method, gp_kernel=gp_kernel),
            X_sub,
            Y_sub,
            cv=min(cv, size),
            shuffle=shuffle,
            random_state=random_state,
            verbose=False,
        )

        for name in outputs:
            r2_per_size[name].append(cv_report[name]["r2_mean"])
            r2_std_per_size[name].append(cv_report[name]["r2_std"])
            rmse_per_size[name].append(cv_report[name]["rmse_mean"])
            rmse_std_per_size[name].append(cv_report[name]["rmse_std"])

        if verbose:
            r2_str = ", ".join(f"{n}={r2_per_size[n][-1]:.4f}" for n in outputs)
            print(f"Size {size:4d}: {r2_str}")

    result: dict[str, Any] = {"train_sizes": train_sizes}
    for name in outputs:
        result[name] = {
            "r2_mean": np.array(r2_per_size[name]),
            "r2_std": np.array(r2_std_per_size[name]),
            "rmse_mean": np.array(rmse_per_size[name]),
            "rmse_std": np.array(rmse_std_per_size[name]),
        }

    return result
