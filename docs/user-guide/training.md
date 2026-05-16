# Training

## Choosing a backend

| Backend | Method | Strengths | Limitations |
|---------|--------|-----------|-------------|
| **Gaussian Process** | `method="gp"` | Uncertainty estimates, excellent with small data | O(n³) scaling, no ONNX export |
| **MLP** | `method="mlp"` | Scales to large data, ONNX export | Needs tuning, no built-in uncertainty |

## Gaussian Process

Good for small datasets (< 1000 points) where you want prediction uncertainty.

```python
surrogate = pr.Surrogate(
    params=["alpha", "camber"],
    outputs=["CL"],
    method="gp"
)
surrogate.fit(X, Y)
```

GP models use GPyTorch's ExactGP. They automatically optimize kernel hyperparameters.

## MLP

Good for larger datasets or when you need ONNX deployment.

```python
surrogate = pr.Surrogate(
    params=["alpha", "mach"],
    outputs=["CL", "CD"],
    method="mlp"
)
surrogate.fit(X, Y, n_iter=3000, lr=0.001)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_iter` | 1000 | Number of training iterations |
| `lr` | 0.001 | Learning rate |
| `physics` | None | List of physics constraints |

## Multi-output

One surrogate predicts multiple outputs simultaneously:

```python
def my_airfoil(alpha, mach):
    return {"CL": ..., "CD": ..., "CM": ...}

surrogate = pr.Surrogate(
    params=["alpha", "mach"],
    outputs=["CL", "CD", "CM"],
    method="gp"
)
surrogate.fit(X, Y)
```

Each output gets its own independent GP or MLP head.
