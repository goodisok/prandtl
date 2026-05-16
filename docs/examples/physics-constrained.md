# Example: Physics-Constrained Training

Enforcing monotonicity, boundary values, and convexity simultaneously.

```python
import prandtl as pr
from prandtl import Monotonicity, BoundaryValue, Convexity
import numpy as np

# Generate data from a physically plausible function
def airfoil_data(alpha, mach):
    cl = 2 * np.pi * (np.radians(alpha) + 0.04)
    cd = 0.01 + 0.1 * cl**2
    return {"CL": cl, "CD": cd}

X, Y = pr.sample(airfoil_data, bounds=[(-5, 15), (0.15, 0.85)], n=200, seed=42)

# Define physics
constraints = [
    Monotonicity(param_idx=0, sign=1, weight=0.1),
    # CL MUST increase with alpha
    BoundaryValue({"alpha": 0.0}, {"CL": 0.0}, weight=10.0),
    # CL=0 at zero angle of attack
    Convexity(param_idx=0, sign=-1, weight=0.05),
    # Drag polar is concave
]

surrogate = pr.Surrogate(
    params=["alpha", "mach"],
    outputs=["CL", "CD"],
    method="mlp"
)
surrogate.fit(X, Y, physics=constraints, n_iter=500, lr=0.01)

# Verify physics: CL at alpha=0 should be ~0
X_check = np.array([[0.0, 0.5]])  # alpha=0, mach=0.5
cl_at_zero = surrogate.predict(X_check)["CL"][0]
print(f"CL at α=0: {cl_at_zero:.6f}")  # should be very close to 0

# Verify monotonicity: CL should increase with alpha
X_low = np.array([[0.0, 0.5]])
X_high = np.array([[10.0, 0.5]])
cl_low = surrogate.predict(X_low)["CL"][0]
cl_high = surrogate.predict(X_high)["CL"][0]
print(f"CL(α=0)={cl_low:.4f} < CL(α=10)={cl_high:.4f}: {cl_low < cl_high}")
```
