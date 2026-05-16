# Example: Thin Airfoil Theory

Using analytical truth to validate the surrogate pipeline — zero CFD required.

```python
import prandtl as pr

# Thin airfoil: CL = 2π(α + 2 camber)
X, Y = pr.sample(
    pr.analytical.cl_flat_plate,
    bounds=[(-5, 15), (0.01, 0.1)],
    n=100, method="lhs", seed=42
)

surrogate = pr.Surrogate(
    params=["alpha", "camber"],
    outputs=["CL"],
    method="gp"
)
surrogate.fit(X, Y)

# Test on held-out points
X_test, Y_test = pr.sample(
    pr.analytical.cl_flat_plate,
    bounds=[(-5, 15), (0.01, 0.1)],
    n=30, seed=99
)
Y_pred = surrogate.predict(X_test)

# Evaluate
report = pr.metrics(Y_test, Y_pred)
print(f"R²: {report['CL']['r2']:.6f}")  # > 0.9999

# Cross-validate
scores = pr.cross_validate(surrogate, X, Y, cv=5)
print(f"CV MAE: {scores['CL']['mae_mean']:.6f} ± {scores['CL']['mae_std']:.6f}")
```

Smooth analytical functions are nearly perfectly learned by GP with just 100 points.
