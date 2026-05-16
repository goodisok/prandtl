# Quick Start

Prandtl follows a simple 3-step workflow: **sample → fit → evaluate**.

## 1. Sample the parameter space

```python
import prandtl as pr

X, Y = pr.sample(
    pr.analytical.cl_flat_plate,
    bounds=[(-5, 15), (0.01, 0.1)],
    n=100, method="lhs", seed=42
)
```

`X` is a (100, 2) array of [alpha, camber] values.  
`Y` is a (100, 1) array of lift coefficients.

## 2. Train a surrogate

```python
surrogate = pr.Surrogate(
    params=["alpha", "camber"],
    outputs=["CL"],
    method="gp"
)
surrogate.fit(X, Y)
```

Choose `method="gp"` for small data (< 1000 points, uncertainty estimates).  
Choose `method="mlp"` for large data or ONNX deployment.

## 3. Predict and evaluate

```python
Y_pred = surrogate.predict(X_test)
report = pr.metrics(Y_test, Y_pred)
print(report)
# {"CL": {"r2": 0.9998, "rmse": 0.0012, "mae": 0.0010, ...}}
```

## Going deeper

- [Sampling methods](sampling.md) — LHS, Sobol, uniform
- [Training options](training.md) — GP vs MLP, hyperparameters
- [Validation suite](validation.md) — cross-validation, learning curves, residuals
- [Physics constraints](physics.md) — inject domain knowledge
- [ONNX export](export.md) — deploy anywhere
- [CFD data I/O](io.md) — read OpenFOAM and SU2 output
