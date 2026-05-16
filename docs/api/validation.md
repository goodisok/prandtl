# Validation Functions

Metrics and diagnostics for surrogate evaluation.

## metrics()

```python
from prandtl import metrics
report = metrics(Y_true, Y_pred)
# → {"CL": {"r2": 0.999, "rmse": 0.001, "mae": 0.001, ...}}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `Y_true` | dict[str, ndarray] | Ground truth, keyed by output name |
| `Y_pred` | dict[str, ndarray] | Predictions, same keys |

| Return metric | Description |
|---------------|-------------|
| `r2` | Coefficient of determination (1 = perfect) |
| `rmse` | Root mean square error |
| `mae` | Mean absolute error |
| `max_re` | Maximum residual error |
| `explained_variance` | Explained variance score |

## cross_validate()

```python
from prandtl import cross_validate
scores = cross_validate(surrogate, X, Y, cv=5)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `surrogate` | Surrogate | — | Trained surrogate |
| `X` | ndarray | — | Input data |
| `Y` | dict | — | Output data |
| `cv` | int | 5 | Number of folds |
| `verbose` | bool | False | Print progress |

Returns per-output `mae_mean` and `mae_std` across folds.

## residual_analysis()

```python
from prandtl import residual_analysis
res = residual_analysis(Y_true, Y_pred)
```

Returns per-output: `shapiro_p` (normality), `skewness`, `kurtosis`, `max_residual_idx`.

## learning_curve()

```python
from prandtl import learning_curve
curve = learning_curve(surrogate, X, Y, sizes=[20, 50, 100, 200])
```

Returns `train_sizes`, `train_mae`, `val_mae` for each training size.
