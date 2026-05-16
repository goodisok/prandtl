# Example: Full Validation Pipeline

End-to-end validation workflow — from training to deployment confidence.

```python
import prandtl as pr
from prandtl import cross_validate, learning_curve, residual_analysis, metrics

# 1. Generate data
X, Y = pr.sample(
    pr.analytical.cl_flat_plate,
    bounds=[(-5, 15), (0.01, 0.1)],
    n=200, method="lhs", seed=42
)

# 2. Train
surrogate = pr.Surrogate(params=["alpha", "camber"], outputs=["CL"], method="gp")
surrogate.fit(X, Y)

# 3. Cross-validate
cv = cross_validate(surrogate, X, Y, cv=5)
print(f"CV MAE: {cv['CL']['mae_mean']:.4f} ± {cv['CL']['mae_std']:.4f}")

# 4. Learning curve
curve = learning_curve(surrogate, X, Y, sizes=[20, 50, 100, 150, 200])
print("Sizes:", curve["train_sizes"])
print("Train MAE:", [f"{x:.4f}" for x in curve["train_mae"]])
print("Val MAE:  ", [f"{x:.4f}" for x in curve["val_mae"]])
# If val_mae stops improving → enough data

# 5. Residual diagnostics
X_test, Y_test = pr.sample(
    pr.analytical.cl_flat_plate,
    bounds=[(-5, 15), (0.01, 0.1)],
    n=50, seed=99
)
Y_pred = surrogate.predict(X_test)
res = residual_analysis(Y_test, Y_pred)

for output in res:
    r = res[output]
    is_normal = r["shapiro_p"] > 0.05
    print(f"Shapiro-Wilk p={r['shapiro_p']:.3f} → {'normal ✓' if is_normal else 'non-normal ✗'}")
    print(f"  Skewness: {r['skewness']:.3f}")
    print(f"  Kurtosis: {r['kurtosis']:.3f}")

# 6. Final metrics
report = metrics(Y_test, Y_pred)
print(f"Final R²: {report['CL']['r2']:.4f}")
print(f"Final MAE: {report['CL']['mae']:.4f}")
```

## Interpreting the results

| Check | Good sign | Warning sign |
|-------|-----------|--------------|
| CV MAE | Stable across folds | High variance → uneven data |
| Learning curve | val_mae plateaus | Still improving → need more data |
| Shapiro-Wilk | p > 0.05 | p < 0.05 → systematic bias |
| Skewness | ~0 | > ±0.5 → directional bias |
