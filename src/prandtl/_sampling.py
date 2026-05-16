"""Parameter space sampling for surrogate model training.

Provides Latin Hypercube Sampling (LHS), uniform random, and Sobol sequence
methods for generating design points in parameter space.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

import numpy as np
from scipy.stats import qmc


def sample(
    func: Callable[..., dict[str, float]],
    bounds: list[tuple[float, float]],
    n: int,
    *,
    method: str = "lhs",
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a parameter space and evaluate a truth function at each point.

    Parameters
    ----------
    func : callable
        Truth function that accepts keyword arguments matching parameter names
        and returns a dict of output values. Example: ``lambda alpha, camber: {"CL": 2*pi*(alpha+2*camber)}``.
    bounds : list of (float, float)
        Lower and upper bounds for each parameter, in order.
    n : int
        Number of design points to generate.
    method : str
        Sampling strategy: ``'lhs'`` (Latin Hypercube, default), ``'uniform'``
        (uniform random), or ``'sobol'`` (Sobol quasi-random sequence).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n, d)
        Design matrix — each row is a parameter vector.
    Y : ndarray of shape (n, k)
        Output matrix — each row is the truth function outputs for the
        corresponding parameter vector. Output columns are sorted alphabetically
        by key name.
    """
    if seed is not None:
        np.random.seed(seed)

    d = len(bounds)
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)

    # Generate unit-cube samples
    if method == "lhs":
        sampler = qmc.LatinHypercube(d=d, seed=seed)
        unit = sampler.random(n=n)
    elif method == "sobol":
        sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
        unit = sampler.random(n=n)
    elif method == "uniform":
        unit = np.random.uniform(size=(n, d))
    else:
        raise ValueError(f"Unknown sampling method: {method!r}. Use 'lhs', 'uniform', or 'sobol'.")

    # Scale to bounds
    X = lo + unit * (hi - lo)

    # Evaluate truth function
    Y_rows = []
    # Preserve parameter order from function signature (matches bounds order)
    sig = inspect.signature(func)
    param_keys = list(sig.parameters.keys())
    if len(param_keys) != d:
        raise ValueError(
            f"Function has {len(param_keys)} parameters but {d} bounds were provided. "
            f"Parameter names: {param_keys}. Ensure bounds are given in the same order "
            f"as the function signature."
        )
    for i in range(n):
        kwargs = dict(zip(param_keys, X[i].tolist()))
        result = func(**kwargs)
        # Ensure consistent column order
        sorted_keys = sorted(result.keys())
        Y_rows.append([result[k] for k in sorted_keys])

    return X, np.array(Y_rows, dtype=np.float64)
