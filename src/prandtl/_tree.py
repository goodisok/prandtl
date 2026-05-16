"""Tree-based regression backends: Random Forest and Gradient Boosting.

These models complement the Gaussian Process and neural network backends
in ``_surrogate.py``. They are provided as standalone classes for users
who prefer a scikit-learn-style interface.
"""

from __future__ import annotations

import numpy as np


class RandomForest:
    """Random Forest regressor with uncertainty via tree variance.

    Parameters
    ----------
    n_estimators : int
        Number of trees (default: 200).
    max_depth : int or None
        Maximum tree depth (default: None = unlimited).
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self._model = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> RandomForest:
        """Fit the random forest.

        Parameters
        ----------
        X : (n, d) training inputs
        Y : (n,) or (n, 1) training targets
        """
        from sklearn.ensemble import RandomForestRegressor

        Y = np.asarray(Y, dtype=np.float64).ravel()
        X = np.asarray(X, dtype=np.float64)

        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(X, Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict mean values."""
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        return self._model.predict(np.asarray(X, dtype=np.float64))

    def predict_with_uncertainty(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return predictive mean and standard deviation (tree variance)."""
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        X = np.asarray(X, dtype=np.float64)

        # Collect predictions from all trees
        tree_preds = np.array(
            [tree.predict(X) for tree in self._model.estimators_]
        )  # (n_trees, n_samples)

        mean = tree_preds.mean(axis=0)
        std = tree_preds.std(axis=0, ddof=1)
        return mean, std


class GradientBoosting:
    """Gradient Boosting regressor.

    Parameters
    ----------
    n_estimators : int
        Number of boosting stages (default: 200).
    max_depth : int
        Maximum tree depth (default: 5).
    learning_rate : float
        Shrinkage parameter (default: 0.05).
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self._model = None
        self._lower_model = None
        self._upper_model = None

    def fit(self, X: np.ndarray, Y: np.ndarray) -> GradientBoosting:
        """Fit the gradient boosting model."""
        from sklearn.ensemble import GradientBoostingRegressor

        Y = np.asarray(Y, dtype=np.float64).ravel()
        X = np.asarray(X, dtype=np.float64)

        self._model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )
        self._model.fit(X, Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict mean values."""
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        return self._model.predict(np.asarray(X, dtype=np.float64))

    def fit_with_uncertainty(
        self, X: np.ndarray, Y: np.ndarray, alpha: float = 0.05
    ) -> GradientBoosting:
        """Fit three quantile regressors for uncertainty quantification.

        Trains models for the 50th percentile (median) and the lower/upper
        quantiles defined by alpha.

        Parameters
        ----------
        X : (n, d) training inputs
        Y : (n,) training targets
        alpha : float
            Significance level. The lower bound is alpha/2,
            the upper bound is 1 - alpha/2. Default 0.05 gives 95% PI.
        """
        from sklearn.ensemble import GradientBoostingRegressor

        Y = np.asarray(Y, dtype=np.float64).ravel()
        X = np.asarray(X, dtype=np.float64)

        self._lower_model = GradientBoostingRegressor(
            loss="quantile", alpha=alpha / 2,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )
        self._lower_model.fit(X, Y)

        self._model = GradientBoostingRegressor(
            loss="quantile", alpha=0.5,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )
        self._model.fit(X, Y)

        self._upper_model = GradientBoostingRegressor(
            loss="quantile", alpha=1 - alpha / 2,
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.random_state,
        )
        self._upper_model.fit(X, Y)
        return self

    def predict_with_uncertainty(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return median prediction and uncertainty (requires fit_with_uncertainty)."""
        if self._lower_model is None:
            raise RuntimeError(
                "Call fit_with_uncertainty() first to enable uncertainty estimation."
            )
        X = np.asarray(X, dtype=np.float64)
        lower = self._lower_model.predict(X)
        median = self._model.predict(X)
        upper = self._upper_model.predict(X)
        # Approximate std from quantile range
        std = (upper - lower) / 3.92  # 95% CI → σ
        return median, std
