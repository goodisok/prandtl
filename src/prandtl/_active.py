"""Active learning and Bayesian optimisation for surrogate models.

Use predictive uncertainty to guide sampling — automatically discover where
to place the next datapoint for maximum information gain.

Strategies
----------
* ``'ei'`` — Expected Improvement (for minimisation problems)
* ``'ucb'`` — Upper Confidence Bound (exploration–exploitation tradeoff)
* ``'pi'`` — Probability of Improvement
* ``'uncertainty'`` — pure uncertainty sampling (maximise predictive variance)
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

import numpy as np
import torch
from scipy.optimize import minimize


def propose_next(
    surrogate,
    bounds: list[tuple[float, float]],
    *,
    strategy: str = "uncertainty",
    beta: float = 2.0,
    n_restarts: int = 20,
    y_best: float | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Propose the next point to sample for maximum information gain.

    Optimises an acquisition function over the bounded parameter space using
    multi-start L-BFGS-B. Only works with ``method='gp'`` surrogates (MLP
    does not provide uncertainty estimates).

    Parameters
    ----------
    surrogate : Surrogate
        A **fitted** GP surrogate model.
    bounds : list of (float, float)
        Lower and upper bounds for each parameter, in order.
    strategy : str
        Acquisition function:
        ``'uncertainty'`` — maximise predictive variance (default)
        ``'ei'`` — Expected Improvement (minimisation)
        ``'ucb'`` — Upper Confidence Bound
        ``'pi'`` — Probability of Improvement
    beta : float
        Exploration–exploitation tradeoff for ``'ucb'``. Higher = more
        exploration. Default 2.0.
    n_restarts : int
        Number of random restarts for the optimiser. Higher = more thorough
        search at the cost of computation time.
    y_best : float or None
        Best observed value so far. Required for ``'ei'`` and ``'pi'``.
        If None for those strategies, it is computed from the surrogate's
        training data using a fallback mean prediction.
    seed : int or None
        Random seed for restart points.

    Returns
    -------
    ndarray of shape (n_params,)
        The proposed next sampling point.
    """
    if surrogate._method != "gp":
        raise ValueError(
            "Active learning requires method='gp' for uncertainty estimates. "
            f"Got method={surrogate._method!r}."
        )

    rng = np.random.RandomState(seed)
    d = len(bounds)
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)

    # Determine y_best for EI/PI strategies
    if strategy in ("ei", "pi") and y_best is None:
        # Fallback: use the minimum of the training data
        # This is approximate — for best results, track y_best externally
        y_best = _infer_y_best(surrogate)

    # Acquisition function
    def _acq(x: np.ndarray) -> float:
        """Negative acquisition (we minimise, acquisition is to maximise)."""
        x_2d = x.reshape(1, -1)
        mean, std = surrogate.predict_with_uncertainty(x_2d)
        mean_val = float(mean[0, 0])
        std_val = float(std[0, 0])

        if strategy == "uncertainty":
            acq_val = std_val**2  # variance

        elif strategy == "ei":
            if std_val < 1e-12:
                acq_val = 0.0
            else:
                delta = y_best - mean_val  # improvement (minimisation)
                z = delta / std_val
                from scipy.stats import norm
                acq_val = delta * norm.cdf(z) + std_val * norm.pdf(z)

        elif strategy == "pi":
            if std_val < 1e-12:
                acq_val = 0.0
            else:
                from scipy.stats import norm
                z = (y_best - mean_val) / std_val
                acq_val = norm.cdf(z)

        elif strategy == "ucb":
            acq_val = mean_val + beta * std_val

        else:
            raise ValueError(
                f"Unknown strategy: {strategy!r}. "
                f"Use 'uncertainty', 'ei', 'ucb', or 'pi'."
            )

        return -float(acq_val)  # negative for minimisation

    # Multi-start L-BFGS-B
    best_x = None
    best_acq = np.inf

    for _ in range(n_restarts):
        x0 = lo + rng.uniform(size=d) * (hi - lo)
        result = minimize(
            _acq,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 50, "ftol": 1e-8},
        )
        if result.fun < best_acq:
            best_acq = result.fun
            best_x = result.x

    return np.asarray(best_x if best_x is not None else lo, dtype=np.float64)


def active_learn(
    func: Callable[..., dict[str, float]],
    bounds: list[tuple[float, float]],
    surrogate,
    *,
    n_initial: int = 10,
    n_iter: int = 10,
    strategy: str = "uncertainty",
    beta: float = 2.0,
    seed: int | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Active learning loop: iteratively sample where the model is most uncertain.

    Starts with ``n_initial`` random Latin Hypercube samples, then for
    ``n_iter`` iterations: proposes the next point via the acquisition
    strategy, evaluates the truth function, and refits the surrogate.

    Parameters
    ----------
    func : callable
        Ground-truth function. Accepts keyword arguments matching the
        parameter names in ``surrogate._params`` and returns a dict of
        output values.
    bounds : list of (float, float)
        Parameter bounds.
    surrogate : Surrogate
        An **unfitted** GP surrogate instance used as a configuration
        template (params, outputs, kernel, etc.).
    n_initial : int
        Number of initial random samples before active learning begins.
    n_iter : int
        Number of active learning iterations.
    strategy : str
        Acquisition strategy: ``'uncertainty'``, ``'ei'``, ``'ucb'``, or ``'pi'``.
    beta : float
        UCB exploration parameter.
    seed : int or None
        Random seed for reproducibility.
    verbose : bool
        If True, print progress each iteration.

    Returns
    -------
    X_all : ndarray of shape (n_initial + n_iter, n_params)
        All sampled points.
    Y_all : ndarray of shape (n_initial + n_iter, n_outputs)
        All truth function evaluations.
    history : list of float
        Acquisition function values at each active learning iteration.
    """
    if surrogate._method != "gp":
        raise ValueError(
            "Active learning requires method='gp' for uncertainty estimates."
        )

    from prandtl._sampling import sample
    from prandtl._surrogate import Surrogate

    rng = np.random.RandomState(seed)
    d = len(bounds)

    # Phase 1: initial random sampling
    X_all, Y_all = sample(func, bounds=bounds, n=n_initial, method="lhs", seed=seed)

    # Phase 2: active learning loop
    param_keys = list(inspect.signature(func).parameters.keys())
    history: list[float] = []

    for iteration in range(n_iter):
        # Fit surrogate on current data
        clone = Surrogate(
            params=list(surrogate._params),
            outputs=list(surrogate._outputs),
            method="gp",
            gp_kernel=surrogate._gp_kernel,
        )
        clone.fit(X_all, Y_all, verbose=False)

        # Propose next point
        x_next = propose_next(
            clone,
            bounds,
            strategy=strategy,
            beta=beta,
            seed=seed + iteration if seed else None,
        )

        # Evaluate truth function
        kwargs = dict(zip(param_keys, x_next.tolist()))
        result = func(**kwargs)
        y_next = np.array([result[k] for k in result.keys()], dtype=np.float64)

        # Append
        X_all = np.vstack([X_all, x_next.reshape(1, -1)])
        Y_all = np.vstack([Y_all, y_next.reshape(1, -1)])

        # Track acquisition value
        mean, std = clone.predict_with_uncertainty(x_next.reshape(1, -1))
        if strategy == "uncertainty":
            acq_val = float(std[0, 0] ** 2)
        elif strategy == "ucb":
            acq_val = float(mean[0, 0] + beta * std[0, 0])
        elif strategy in ("ei", "pi"):
            y_best = np.min(Y_all[:, 0]) if Y_all.shape[1] > 0 else 0.0
            delta = y_best - float(mean[0, 0])
            if strategy == "ei" and std[0, 0] > 1e-12:
                from scipy.stats import norm
                z = delta / float(std[0, 0])
                acq_val = float(delta * norm.cdf(z) + std[0, 0] * norm.pdf(z))
            elif strategy == "pi" and std[0, 0] > 1e-12:
                from scipy.stats import norm
                acq_val = float(norm.cdf(delta / float(std[0, 0])))
            else:
                acq_val = 0.0
        else:
            acq_val = 0.0
        history.append(acq_val)

        if verbose:
            print(
                f"Active iter {iteration + 1}/{n_iter}: "
                f"x={x_next[0]:.3f}" + (f" ..." if d > 1 else "") +
                f", acq={acq_val:.4f}"
            )

    return X_all, Y_all, history


def _infer_y_best(surrogate) -> float:
    """Fallback: estimate best observed value from training residuals."""
    # Approximate: find the minimum prediction amongst interpolated training points
    X_train = surrogate._x_mean.reshape(1, -1)  # just centre — very approximate
    mean, _ = surrogate.predict_with_uncertainty(X_train)
    return float(np.min(mean))
