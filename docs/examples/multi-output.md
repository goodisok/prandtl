# Example: Multi-Output Surrogate

Predicting CL and CD simultaneously from alpha and Mach number.

```python
import prandtl as pr
import numpy as np

def my_airfoil(alpha, mach):
    cl = 2 * np.pi * (np.radians(alpha) + 0.04)
    cd = 0.01 + 0.1 * cl**2  # quadratic drag polar
    return {"CL": cl, "CD": cd}

X, Y = pr.sample(
    my_airfoil,
    bounds=[(-5, 15), (0.15, 0.85)],
    n=200, method="lhs", seed=42
)

surrogate = pr.Surrogate(
    params=["alpha", "mach"],
    outputs=["CL", "CD"],
    method="mlp"
)
surrogate.fit(X, Y, n_iter=3000)

# Test
X_test, Y_test = pr.sample(my_airfoil, bounds=[(-5, 15), (0.15, 0.85)], n=50, seed=99)
report = pr.metrics(Y_test, surrogate.predict(X_test))

for output in report:
    r = report[output]
    print(f"{output}: R²={r['r2']:.4f}, MAE={r['mae']:.4f}, RMSE={r['rmse']:.4f}")
```

!!! tip
    Each output gets its own independent model head — no assumption about output correlations.
