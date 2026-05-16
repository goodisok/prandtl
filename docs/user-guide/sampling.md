# Sampling

Prandtl provides three sampling methods for exploring parameter spaces.

## Methods

### Latin Hypercube Sampling (LHS) — default

Space-filling design. Maximizes coverage with minimal points. Best for surrogate modeling.

```python
X, Y = pr.sample(
    pr.analytical.cl_flat_plate,
    bounds=[(-5, 15), (0.01, 0.1)],
    n=100,
    method="lhs",
    seed=42
)
```

### Uniform Random

Simple, unbiased. Useful for validation sets.

```python
X, Y = pr.sample(
    func, bounds=[(0, 1), (-2, 2)],
    n=100, method="uniform", seed=42
)
```

### Sobol Sequence

Low-discrepancy quasi-random. Deterministic — same input always gives same points.

```python
X, Y = pr.sample(
    func, bounds=[(0, 1), (-2, 2)],
    n=128, method="sobol"
)
```

Requires `n` to be a power of 2 for best uniformity.

## Using existing data

You don't need `sample()` if you already have data:

```python
surrogate = pr.Surrogate(params=["alpha", "mach"], outputs=["CL"], method="gp")
surrogate.fit(X, Y)  # X: (n_points, n_params), Y: (n_points, n_outputs)
```

## API

| Parameter | Type | Description |
|-----------|------|-------------|
| `func` | callable | Takes dict of param values → returns dict of output values |
| `bounds` | list of (min, max) | One tuple per parameter |
| `n` | int | Number of samples |
| `method` | str | `"lhs"`, `"uniform"`, or `"sobol"` |
| `seed` | int | Random seed for reproducibility |
