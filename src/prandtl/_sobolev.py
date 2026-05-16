"""Sobolev training for Gaussian Process surrogate models.

Sobolev training incorporates gradient information into GP fitting,
improving accuracy with fewer function evaluations. This is particularly
valuable in CFD where adjoint solvers can provide cheap gradient data.

The gradient of a GP posterior mean is computed analytically via kernel
derivatives, enabling both gradient-constrained fitting and gradient
prediction at new points.

Reference
---------
Cressie, N. (1993). *Statistics for Spatial Data*. Wiley.
"""

from __future__ import annotations

import numpy as np
import torch


def predict_gradient(
    model,  # _ExactGP
    train_x: torch.Tensor,
    x_query: torch.Tensor,
) -> np.ndarray:
    """Analytic gradient of GP posterior mean w.r.t. inputs.

    For an RBF kernel: k(x, x') = σ² exp(-½||x-x'||² / l²)

    The posterior mean is μ(x*) = k(x*, X)·α  where α = K⁻¹·y

    Gradient: ∂μ/∂x*_j = Σᵢ αᵢ · ∂k(x*, xᵢ)/∂x*_j
                       = Σᵢ αᵢ · (-(x*_j - xᵢ_j)/l²) · k(x*, xᵢ)

    Parameters
    ----------
    model : _ExactGP
        A trained GP model.
    train_x : Tensor of shape (n, d)
        Training inputs.
    x_query : Tensor of shape (m, d)
        Query points.

    Returns
    -------
    ndarray of shape (m, d)
        Predicted gradients.
    """
    model.eval()
    model.likelihood.eval()

    with torch.no_grad():
        return _compute_predicted_gradient(model, train_x, model.train_targets, x_query)


def fit_with_gradients(
    X: np.ndarray,
    Y: np.ndarray,
    dY_dX: np.ndarray,
    *,
    n_iter: int = 300,
    lr: float = 0.05,
    grad_weight: float = 0.5,
    verbose: bool = True,
) -> tuple:
    """Fit a GP with Sobolev gradient constraints.

    Minimises a combined loss:
        L = (1 - α) · L_MLL + α · L_gradient

    This is an experimental feature exposed as a standalone function.

    Parameters
    ----------
    X : ndarray of shape (n, d)
        Training inputs.
    Y : ndarray of shape (n,)
        Training outputs.
    dY_dX : ndarray of shape (n, d)
        Observed gradients at each training point.
    n_iter : int
        Number of optimisation steps.
    lr : float
        Learning rate.
    grad_weight : float
        Weight of gradient loss in combined objective.
    verbose : bool
        Print progress.

    Returns
    -------
    model : _ExactGP
        Trained GP model (call .eval() before prediction).
    x_mean, x_std, y_mean, y_std : ndarray
        Standardisation parameters.
    """
    from prandtl._gaussian import _ExactGP

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64).ravel()
    dY_dX = np.asarray(dY_dX, dtype=np.float64)

    if X.shape[0] != dY_dX.shape[0]:
        raise ValueError(
            f"X and dY_dX must have same number of samples: {X.shape[0]} vs {dY_dX.shape[0]}"
        )
    if X.shape[1] != dY_dX.shape[1]:
        raise ValueError(
            f"X and dY_dX must have same input dimension: {X.shape[1]} vs {dY_dX.shape[1]}"
        )

    # Standardise
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0, ddof=0)
    x_std[x_std < 1e-12] = 1.0
    y_mean = float(Y.mean())
    y_std = float(Y.std(ddof=0))
    if y_std < 1e-12:
        y_std = 1.0

    X_s = (X - x_mean) / x_std
    Y_s = (Y - y_mean) / y_std
    dY_dX_s = dY_dX * x_std[None, :] / y_std

    x_t = torch.tensor(X_s, dtype=torch.float32)
    y_t = torch.tensor(Y_s, dtype=torch.float32)
    grad_t = torch.tensor(dY_dX_s, dtype=torch.float32)

    model = _ExactGP(x_t, y_t, kernel="rbf")
    model.train()
    model.likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    import gpytorch
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    alpha_loss = grad_weight
    for i in range(n_iter):
        optimizer.zero_grad()

        output = model(x_t)
        gp_loss = -mll(output, y_t)

        # Compute predicted gradients analytically
        pred_grad = _compute_predicted_gradient(
            model, x_t, y_t, x_t
        )
        grad_loss = torch.nn.functional.mse_loss(pred_grad, grad_t)

        loss = (1.0 - alpha_loss) * gp_loss + alpha_loss * grad_loss
        loss.backward()
        optimizer.step()

        if verbose and i % 100 == 0:
            print(
                f"  Sobolev iter {i:4d}/{n_iter}  "
                f"gp_loss={gp_loss.item():.4f}  grad_loss={grad_loss.item():.4f}"
            )

    model.eval()
    model.likelihood.eval()
    return model, x_mean, x_std, y_mean, y_std


def _compute_predicted_gradient(
    model, train_x: torch.Tensor, train_y: torch.Tensor, x_query: torch.Tensor
) -> torch.Tensor:
    """Compute analytic gradient of GP posterior mean at query points.

    Returns tensor with requires_grad for backprop through hyperparameters.
    """
    outputscale = model.covar_module.outputscale
    base_kernel = model.covar_module.base_kernel
    if hasattr(base_kernel, "lengthscale"):
        lengthscale = base_kernel.lengthscale
    else:
        lengthscale = torch.ones(train_x.shape[1])

    n_train, d = train_x.shape
    n_query = x_query.shape[0]

    # Pairwise kernel
    diff = x_query.unsqueeze(1) - train_x.unsqueeze(0)  # (m, n, d)

    if lengthscale.shape[-1] == 1:
        ls_sq = lengthscale**2
        scaled_sq = -0.5 * (diff**2).sum(-1) / ls_sq  # (m, n)
        K_xX = outputscale * torch.exp(scaled_sq)  # (m, n)
        grad_factor = -diff / ls_sq  # (m, n, d)
    else:
        ls_sq = (lengthscale.squeeze(0))**2  # (d,)
        scaled_sq = -0.5 * ((diff**2) / ls_sq.unsqueeze(0).unsqueeze(0)).sum(-1)  # (m, n)
        K_xX = outputscale * torch.exp(scaled_sq)
        grad_factor = -diff / ls_sq.unsqueeze(0).unsqueeze(0)

    # K(X,X) for alpha
    K_train = model.covar_module(train_x, train_x).evaluate()
    noise = model.likelihood.noise_covar.noise
    K_train = K_train + noise * torch.eye(n_train, device=train_x.device)

    alpha = torch.linalg.solve(K_train, train_y)  # (n,)

    # ∇μ = Σᵢ αᵢ · ∂K/∂x*
    K_weighted = K_xX.unsqueeze(-1) * grad_factor  # (m, n, d)
    grad = (K_weighted * alpha.unsqueeze(0).unsqueeze(-1)).sum(1)  # (m, d)

    return grad


class soboloev:
    """GP surrogate with Sobolev (gradient-constrained) training.

    Sobolev training uses gradient information alongside function values
    to improve accuracy with fewer evaluations. This is valuable in CFD
    where adjoint solvers can provide cheap gradient data.

    The GP posterior mean gradient is computed analytically via kernel
    derivatives — no finite differences needed.

    Parameters
    ----------
    params : list of str
        Input parameter names.
    output : str
        Output variable name.
    kernel : str
        GP kernel: ``'rbf'`` (default) or ``'matern52'``.
    grad_weight : float
        Weight of gradient matching in the combined loss.
        0 = standard GP, 1 = gradient only.
        Default: 0.5.

    Examples
    --------
    >>> import prandtl as pr
    >>> import numpy as np
    >>> # 1D function with known gradient
    >>> def f(x):
    ...     return np.sin(3*x), 3*np.cos(3*x)
    >>> X = np.random.uniform(0, 2, (8, 1))
    >>> Y, dY = f(X)
    >>> mdl = pr.soboloev(params=["x"], output="y")
    >>> mdl.fit(X, Y, dY)
    >>> y_pred, y_std = mdl.predict(np.array([[1.0]]))
    >>> grad_pred = mdl.predict_gradient(np.array([[1.0]]))
    """

    def __init__(
        self,
        params: list,
        output: str,
        kernel: str = "rbf",
        grad_weight: float = 0.5,
    ) -> None:
        if kernel not in ("rbf", "matern52"):
            raise ValueError(f"soboloev kernel must be 'rbf' or 'matern52', got '{kernel}'")
        if not (0 <= grad_weight <= 1):
            raise ValueError("grad_weight must be in [0, 1]")

        self.params = list(params)
        self.output = output
        self.kernel = kernel
        self.grad_weight = grad_weight
        self._model = None
        self._xm = None
        self._xs = None
        self._ym = None
        self._ys = None

    def __repr__(self) -> str:
        return f"soboloev(params={self.params}, output='{self.output}', kernel='{self.kernel}')"

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        dY_dX: np.ndarray,
        *,
        n_iter: int = 300,
        lr: float = 0.05,
        verbose: bool = True,
    ) -> "soboloev":
        """Fit the GP model with Sobolev gradient constraints.

        Parameters
        ----------
        X : ndarray of shape (n, d)
            Training inputs.
        Y : ndarray of shape (n,) or (n, 1)
            Training outputs.
        dY_dX : ndarray of shape (n, d)
            Gradients at each training point: ∂y/∂x_j at x_i.
        n_iter : int
            Optimisation steps.
        lr : float
            Adam learning rate.
        verbose : bool
            Print progress.

        Returns
        -------
        self
        """
        self._model, self._xm, self._xs, self._ym, self._ys = fit_with_gradients(
            X, Y, dY_dX,
            n_iter=n_iter, lr=lr, grad_weight=self.grad_weight, verbose=verbose,
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict output at query points.

        Parameters
        ----------
        X : ndarray of shape (m, d)

        Returns
        -------
        ndarray of shape (m,)
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        X_s = (np.asarray(X, dtype=np.float64) - self._xm) / self._xs
        with torch.no_grad():
            pred = self._model(torch.tensor(X_s, dtype=torch.float32)).mean.numpy()
        return pred * self._ys + self._ym

    def predict_gradient(self, X: np.ndarray) -> np.ndarray:
        """Predict output gradients at query points.

        Parameters
        ----------
        X : ndarray of shape (m, d)

        Returns
        -------
        ndarray of shape (m, d)
            Predicted gradient ∂y/∂x_j at each query point.
        """
        if self._model is None:
            raise RuntimeError("Call fit() before predict_gradient().")
        X_s = (np.asarray(X, dtype=np.float64) - self._xm) / self._xs
        x_q = torch.tensor(X_s, dtype=torch.float32)
        train_x_t = self._model.train_inputs[0].to(torch.float32)
        grad_s = predict_gradient(self._model, train_x_t, x_q)
        return grad_s * self._ys / self._xs
