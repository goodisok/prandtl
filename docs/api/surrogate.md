# Surrogate

Core model class — train, predict, export.

```python
from prandtl import Surrogate

surrogate = Surrogate(
    params=["alpha", "mach"],
    outputs=["CL", "CD"],
    method="gp"
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `params` | list[str] | — | Input parameter names |
| `outputs` | list[str] | — | Output variable names |
| `method` | str | `"gp"` | `"gp"` or `"mlp"` |

## Methods

### fit(X, Y, **kwargs)

Train the surrogate.

```python
surrogate.fit(X, Y)
surrogate.fit(X, Y, n_iter=500, lr=0.01, physics=constraints)  # MLP opts
```

### predict(X)

Predict on new inputs.

```python
Y_pred = surrogate.predict(X_test)
# → {"CL": array([...]), "CD": array([...])}
```

### validate(X, Y)

Quick validation on a test set.

```python
report = surrogate.validate(X_test, Y_test)
# → {"CL": {"r2": 0.999, "rmse": 0.001, ...}, ...}
```

### export(path)

Export to ONNX (MLP only).

```python
surrogate.export("model")
# → model__CL.onnx, model__CD.onnx
```
