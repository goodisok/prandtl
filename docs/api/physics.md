# Physics Constraints

Inject domain knowledge into MLP training.

## Monotonicity

```python
from prandtl import Monotonicity
Monotonicity(param_idx=0, sign=1, weight=0.1)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `param_idx` | int | — | Which input parameter |
| `sign` | int | — | +1 increasing, -1 decreasing |
| `weight` | float | — | Constraint strength (0.01–1.0) |

## Convexity

```python
from prandtl import Convexity
Convexity(param_idx=0, sign=-1, weight=0.05)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `param_idx` | int | — | Which input parameter |
| `sign` | int | — | +1 convex, -1 concave |
| `weight` | float | — | Constraint strength (0.01–1.0) |

## BoundaryValue

```python
from prandtl import BoundaryValue
BoundaryValue({"alpha": 0.0}, {"CL": 0.0}, weight=10.0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `condition` | dict | — | Parameter values at the boundary |
| `target` | dict | — | Output values at the boundary |
| `weight` | float | — | Constraint strength (1–100) |
