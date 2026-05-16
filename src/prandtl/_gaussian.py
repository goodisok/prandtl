"""Gaussian Process model definitions using GPyTorch.

Internal module — not part of the public API. Used by ``_surrogate.Surrogate``
when ``method='gp'``.
"""

from __future__ import annotations

import gpytorch
import torch


class _ExactGP(gpytorch.models.ExactGP):
    """Exact GP regression with an RBF or Matérn kernel.

    Parameters
    ----------
    train_x : Tensor of shape (n, d)
        Training inputs (already standardized).
    train_y : Tensor of shape (n,)
        Training targets (already standardized).
    kernel : str
        One of ``'rbf'``, ``'matern15'``, ``'matern25'``, ``'matern52'``.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        kernel: str = "rbf",
    ) -> None:
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(train_x, train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()

        # --- Kernel selection ---
        kernel_map = {
            "rbf": gpytorch.kernels.RBFKernel,
            "matern15": lambda: gpytorch.kernels.MaternKernel(nu=1.5),
            "matern25": lambda: gpytorch.kernels.MaternKernel(nu=0.5),
            "matern52": lambda: gpytorch.kernels.MaternKernel(nu=2.5),
        }
        kernel_cls = kernel_map.get(kernel)
        if kernel_cls is None:
            raise ValueError(
                f"Unknown gp_kernel: '{kernel}'. "
                f"Valid options: {list(kernel_map.keys())}."
            )
        base_kernel = kernel_cls() if callable(kernel_cls) else kernel_cls()
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """GP forward pass: mean + covariance at input points."""
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


def _train_gp(
    model: _ExactGP,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    n_iter: int = 200,
    lr: float = 0.1,
    verbose: bool = False,
) -> None:
    """Train an Exact GP model by maximizing marginal log-likelihood.

    Parameters
    ----------
    model : _ExactGP
        The GP model to train.
    train_x, train_y : Tensor
        Training data.
    n_iter : int
        Number of optimization steps.
    lr : float
        Learning rate for the L-BFGS-style optimizer. GPyTorch uses Adam here
        for simplicity; L-BFGS would converge faster but is harder to tune.
    verbose : bool
        If True, print loss every 50 iterations.
    """
    model.train()
    model.likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

        if verbose and i % 50 == 0:
            print(f"  GP iter {i:4d}/{n_iter}  loss={loss.item():.6f}")

    model.eval()
    model.likelihood.eval()
