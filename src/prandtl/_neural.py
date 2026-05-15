"""Neural network surrogate model using PyTorch.

Internal module — not part of the public API. Used by ``_surrogate.Surrogate``
when ``method='mlp'``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _MLP(nn.Module):
    """Three-layer perceptron for regression surrogate modeling.

    A compact architecture suitable for smooth aerodynamic response surfaces.
    Uses ReLU hidden activations and no output activation (regression task).

    Parameters
    ----------
    in_dim : int
        Number of input features (parameter dimensions).
    hidden_dim : int
        Width of each hidden layer. Default 64 is sufficient for most surrogate
        tasks on smooth CFD response surfaces.
    out_dim : int
        Number of outputs. Always 1 — one model is trained per output quantity.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns shape (batch, out_dim)."""
        return self.net(x)


def _train_mlp(
    model: _MLP,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    n_iter: int = 1000,
    lr: float = 0.001,
    verbose: bool = False,
) -> None:
    """Train an MLP surrogate with MSE loss.

    Parameters
    ----------
    model : _MLP
        The network to train.
    train_x : Tensor of shape (n, d)
        Input data.
    train_y : Tensor of shape (n,) or (n, 1)
        Target data.
    n_iter : int
        Number of training epochs.
    lr : float
        Learning rate.
    verbose : bool
        If True, print loss every 100 epochs.
    """
    model.train()

    if train_y.ndim == 1:
        train_y = train_y.unsqueeze(-1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(n_iter):
        optimizer.zero_grad()
        pred = model(train_x)
        loss = criterion(pred, train_y)
        loss.backward()
        optimizer.step()

        if verbose and epoch % 100 == 0:
            print(f"  MLP epoch {epoch:4d}/{n_iter}  loss={loss.item():.6f}")

    model.eval()
