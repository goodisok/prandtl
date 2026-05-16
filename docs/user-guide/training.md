# Training

## Choosing a backend

| Backend | Method | Strengths | Limitations |
|---------|--------|-----------|-------------|
| **Gaussian Process** | `method="gp"` | Uncertainty estimates, excellent with small data | O(n³) scaling, no ONNX export |
| **MLP** | `method="mlp"` | Scales to large data, ONNX export | Needs tuning, no built-in uncertainty |
| **Random Forest** | `method="rf"` | No PyTorch needed, uncertainty via ensemble | Large memory for many trees |
| **Gradient Boosting** | `method="gb"` | High accuracy, no PyTorch | No built-in uncertainty, needs quantile regression |

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

## 树模型 (RF/GB)

Random Forest 和 Gradient Boosting 基于 scikit-learn，**无需 PyTorch/GPyTorch**。适用于中等数据量，训练速度快，内存友好。

```python
# Random Forest
surrogate = pr.Surrogate(
    params=["alpha", "mach"],
    outputs=["CL", "CD"],
    method="rf"
)
surrogate.fit(X, Y)
Y_pred = surrogate.predict(X_test)

# Gradient Boosting
surrogate = pr.Surrogate(
    params=["alpha", "mach"],
    outputs=["CL", "CD"],
    method="gb"
)
surrogate.fit(X, Y)
Y_pred = surrogate.predict(X_test)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | 100 | 树的数量 |
| `max_depth` | None | 树的最大深度 |

!!! tip "RF 不确定性"
    Random Forest 支持 `predict_with_uncertainty()`：
    ```python
    Y_mu, Y_std = surrogate.predict_with_uncertainty(X_test)
    ```
    `Y_std` 是树集成中预测值的标准差，可作为不确定性估计。

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
