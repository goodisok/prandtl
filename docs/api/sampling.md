# Sampling

Parameter space sampling functions.

## sample()

```python
from prandtl import sample
X, Y = sample(func, bounds=[(-5, 15), (0.01, 0.1)], n=100, method="lhs", seed=42)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `func` | callable | — | Function that takes dict of params → dict of outputs |
| `bounds` | list[tuple] | — | (min, max) for each parameter |
| `n` | int | — | Number of samples |
| `method` | str | `"lhs"` | `"lhs"`, `"uniform"`, or `"sobol"` |
| `seed` | int | `None` | Random seed |

| Returns | Type | Description |
|---------|------|-------------|
| `X` | ndarray (n, p) | Parameter samples |
| `Y` | ndarray (n, q) | Output values |

## Methods

| Method | Description | Best for |
|--------|-------------|----------|
| `"lhs"` | Latin Hypercube Sampling | General surrogate modeling |
| `"uniform"` | Uniform random | Validation sets |
| `"sobol"` | Sobol sequence | Deterministic, quasi-random |
