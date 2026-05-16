# Prandtl

**CFD surrogate modeling toolkit.** Train fast aerodynamic surrogates — scikit-learn-like API.

```python
import prandtl as pr

# Sample + learn + predict
X, Y = pr.sample(pr.analytical.cl_flat_plate, bounds=[(-5, 15), (0.01, 0.1)], n=100)
surrogate = pr.Surrogate(params=["alpha", "camber"], outputs=["CL"], method="gp")
surrogate.fit(X, Y)
Y_pred = surrogate.predict(X_test)

# Validate
from prandtl import metrics, cross_validate, learning_curve, residual_analysis
```

## The Problem

CFD simulation: **40 minutes** per run.  
You need **100+** parameter combinations → **66 hours**.

Prandtl: Learn from 100 runs → predict the rest in **milliseconds**, error < 0.2%.

## Key Features

<div class="grid cards" markdown>

- **Zero CFD Required**

    Built-in analytical truth functions — validate your pipeline without touching a solver.

- **Two Backends**

    Gaussian Process (GPyTorch) for small-data regimes.  
    MLP (PyTorch) for large-scale and ONNX export.

- **Validation Suite**

    Cross-validation, learning curves, residual analysis — know if your surrogate actually works.

- **Physics Constraints**

    Monotonicity, convexity, boundary values — inject domain knowledge directly into training.

- **ONNX Export**

    Deploy trained surrogates anywhere: edge devices, real-time control loops, cloud.

- **CFD I/O**

    Parse OpenFOAM forces and SU2 history in one line — from solver output to training-ready.

</div>

## Quick Example

```python
import prandtl as pr
import numpy as np

# 1. Sample the parameter space
X, Y = pr.sample(
    pr.analytical.cl_flat_plate,
    bounds=[(-5, 15), (0.01, 0.1)],
    n=100, method="lhs", seed=42
)

# 2. Train a surrogate
surrogate = pr.Surrogate(
    params=["alpha", "camber"],
    outputs=["CL"],
    method="gp"
)
surrogate.fit(X, Y)

# 3. Predict on new points
X_test, Y_test = pr.sample(
    pr.analytical.cl_flat_plate,
    bounds=[(-5, 15), (0.01, 0.1)],
    n=30, seed=99
)
Y_pred = surrogate.predict(X_test)

# 4. Evaluate
report = pr.metrics(Y_test, Y_pred)
print(report)  # R² > 0.999 on smooth analytical functions
```

[:octicons-arrow-right-24: Get Started](install.md){ .md-button .md-button--primary }
[:octicons-mark-github-24: GitHub](https://github.com/goodisok/prandtl){ .md-button }
