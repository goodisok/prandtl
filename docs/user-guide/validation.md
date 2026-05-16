# Validation

Prandtl provides a comprehensive validation suite to assess surrogate quality.

## Metrics

Basic metrics from predictions vs ground truth:

```python
from prandtl import metrics

report = metrics(Y_test, Y_pred)
# → {"CL": {
#     "r2": 0.9996,           # Coefficient of determination
#     "rmse": 0.0010,         # Root mean square error
#     "mae": 0.0008,          # Mean absolute error
#     "max_re": 0.0034,       # Maximum residual error
#     "explained_variance": 0.9996
# }}
```

## Cross-validation

K-fold cross-validation estimates generalization performance:

```python
from prandtl import cross_validate

scores = cross_validate(surrogate, X, Y, cv=5, verbose=True)

print(f"MAE: {scores['CL']['mae_mean']:.4f} ± {scores['CL']['mae_std']:.4f}")
# → MAE: 0.0123 ± 0.0042
```

| Parameter | Description |
|-----------|-------------|
| `surrogate` | A Surrogate instance (GP or MLP) |
| `X` | Input array (n_points, n_params) |
| `Y` | Output array (n_points, n_outputs) |
| `cv` | Number of folds (default: 5) |
| `verbose` | Print progress per fold |

## Learning Curves

Check if you have enough data:

```python
from prandtl import learning_curve

curve = learning_curve(surrogate, X, Y, sizes=[10, 20, 50, 100, 150])

# Plot interpretation:
# - val_mae plateaus → enough data
# - train_mae << val_mae → overfitting
```

| Key | Description |
|-----|-------------|
| `train_sizes` | Actual sizes used |
| `train_mae` | Training MAE per size |
| `val_mae` | Validation MAE per size |

## Residual Analysis

Diagnose systematic errors:

```python
from prandtl import residual_analysis

res = residual_analysis(Y_test, Y_pred)

for output in res:
    r = res[output]
    print(f"Shapiro-Wilk p={r['shapiro_p']:.3f}")
    print(f"Skewness={r['skewness']:.3f}, Kurtosis={r['kurtosis']:.3f}")
    print(f"Max residual at index {r['max_residual_idx']}")
```

| Metric | What it tells you |
|--------|-------------------|
| `shapiro_p` > 0.05 | Residuals normally distributed ✓ |
| High skewness | Systematic bias — missing physics? |
| High kurtosis | Outliers — bad data points? |
| `max_residual_idx` | Where the worst prediction is |
