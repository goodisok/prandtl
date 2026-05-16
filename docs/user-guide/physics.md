# Physics Constraints

Inject domain knowledge into MLP training — prevent the model from learning physically impossible patterns.

!!! note "MLP only"
    Physics constraints only work with `method="mlp"`. GP models inherently respect smoothness but don't support explicit physical constraints.

## Available Constraints

### Monotonicity

Enforce output monotonicity with respect to a parameter.

```python
from prandtl import Monotonicity

Monotonicity(param_idx=0, sign=1, weight=0.1)
# CL must increase with alpha (param_idx=0). sign=+1 = monotonically increasing.
```

| Parameter | Description |
|-----------|-------------|
| `param_idx` | Which input parameter to constrain |
| `sign` | +1 for increasing, -1 for decreasing |
| `weight` | How strongly to enforce (0.01–1.0) |

### Convexity

Enforce output convexity/concavity.

```python
from prandtl import Convexity

Convexity(param_idx=0, sign=-1, weight=0.05)
# Output is concave w.r.t param_idx=0 (e.g., drag polar curvature)
```

### Boundary Value

Pin output to a known value at specific parameter values.

```python
from prandtl import BoundaryValue

BoundaryValue({"alpha": 0.0}, {"CL": 0.0}, weight=10.0)
# At alpha=0°, CL must equal 0. High weight = strict constraint.
```

| Parameter | Description |
|-----------|-------------|
| `condition` | Dict of parameter values defining the boundary |
| `target` | Dict of output values at that boundary |
| `weight` | Constraint weight (higher = stricter, 1–100) |

## Putting It Together

```python
from prandtl import Monotonicity, BoundaryValue, Convexity

constraints = [
    Monotonicity(param_idx=0, sign=1, weight=0.1),
    BoundaryValue({"alpha": 0.0}, {"CL": 0.0}, weight=10.0),
    Convexity(param_idx=0, sign=-1, weight=0.05),
]

surrogate = pr.Surrogate(
    params=["alpha", "mach"], outputs=["CL", "CD"], method="mlp"
)
surrogate.fit(X, Y, physics=constraints, n_iter=500, lr=0.01)
```

Constraints are applied as soft regularization terms in the loss function — they guide the model toward physically plausible solutions without making convergence impossible.
