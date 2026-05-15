# Prandtl

[English](#prandtl) | [中文](README_zh.md)

CFD surrogate modeling toolkit. Train fast surrogates for aerodynamic predictions — scikit-learn-like API.

```python
import prandtl as pr

# Sample parameter space + analytical truth
X, Y = pr.sample(pr.analytical.cl_flat_plate, bounds=[(-5, 15), (0.01, 0.1)], n=100)

# Train Gaussian Process surrogate
surrogate = pr.Surrogate(params=["alpha", "camber"], outputs=["CL"], method="gp")
surrogate.fit(X, Y)

# Predict + validate
Y_pred = surrogate.predict(X_test)
report = surrogate.validate(X_test, Y_test)
print(report)  # {"CL": {"r2": 0.9998, "rmse": 0.0012, "mae": 0.0010}}

# Export for deployment
surrogate.export("model.onnx")  # one .onnx file per output
```

## The problem

**Question**: How much lift does your drone's rotor generate? → Run a CFD simulation: 40 minutes.

**You actually want 100 different RPM–angle-of-attack combos** → That's 40×100 = 66 hours.

Prandtl's approach: Learn the pattern from 100 sampled points → predict the remaining 10,000 combos in **milliseconds**, error < 0.2%.

Plain English: CFD is an **expensive calculator** — each button press costs 30 minutes. Prandtl **clones that calculator** — the clone returns instant results that are almost indistinguishable from the original.

This is ML at its most practical: not images, not chat, not recommendations — just **learning one function to replace another that's too slow**.

## Install

```bash
pip install prandtl                # Base (numpy, scipy, torch)
pip install prandtl[gp]            # Gaussian Process backend (GPyTorch)
pip install prandtl[export]        # ONNX export support
pip install prandtl[all]           # Everything (gp + export)
```

## v0.2.0 highlights

**Physics constraints** — enforce monotonicity, convexity, or boundary values during MLP training:

```python
from prandtl import Monotonicity, BoundaryValue

surrogate.fit(X, Y, physics=[
    Monotonicity(param_idx=0, sign=1, weight=0.1),
    BoundaryValue({"alpha": 0.0}, {"CL": 0.0}, weight=10.0),
], n_iter=500, lr=0.01)
```

**CFD data I/O** — parse simulation output in one line:

```python
from prandtl import read_foam_forces, read_su2_history

X, Y = read_foam_forces("postProcessing/forces/0/coefficient.dat")
surrogate.fit(X, Y)
```

## What it does

Prandtl lets you replace expensive CFD simulations with fast ML surrogates — without writing any ML boilerplate.

| Feature | Description |
|---------|------------|
| **Zero CFD required** | Validate your surrogate pipeline with built-in analytical truth functions (thin airfoil theory, cylinder drag, propeller thrust) |
| **Two backends** | Gaussian Process (`method='gp'`) via GPyTorch and MLP (`method='mlp'`) via PyTorch |
| **Multi-output** | One surrogate predicts CL, CD, CM simultaneously |
| **Validation reports** | R², RMSE, MAE per output with a single call |
| **ONNX export** | Export trained MLP surrogates for deployment in any ONNX runtime |
| **Sci-kit learn style** | `.fit()`, `.predict()`, `.validate()` — if you know sklearn, you know Prandtl |

## Quick Tour

### 1. Validate with analytical truth (zero CFD)

```python
import prandtl as pr

# Thin airfoil lift coefficient: CL = 2π(α + 2camber)
X, Y = pr.sample(
    pr.analytical.cl_flat_plate,
    bounds=[(-5, 15),    # alpha: -5° to 15°
            (0.01, 0.1)],  # camber: 1% to 10%
    n=100,
    method="lhs",
    seed=42,
)

surrogate = pr.Surrogate(params=["alpha", "camber"], outputs=["CL"], method="gp")
surrogate.fit(X, Y)  # learns the analytical function

# Test on new points
X_test, Y_test = pr.sample(pr.analytical.cl_flat_plate, bounds=[(-5, 15), (0.01, 0.1)], n=30, seed=99)
report = surrogate.validate(X_test, Y_test)
print(report)  # R² > 0.999 on smooth analytical functions
```

### 2. Multiple outputs

```python
def my_airfoil(alpha, mach):
    cl = 2 * np.pi * (np.radians(alpha) + 0.04)
    cd = 0.01 + 0.1 * cl**2  # quadratic drag polar
    return {"CL": cl, "CD": cd}

X, Y = pr.sample(my_airfoil, bounds=[(-5, 15), (0.15, 0.85)], n=200)

surrogate = pr.Surrogate(
    params=["alpha", "mach"], outputs=["CL", "CD"], method="mlp"
)
surrogate.fit(X, Y, n_iter=3000)

# Single call validates all outputs
report = surrogate.validate(X_test, Y_test)
# {"CL": {"r2": 0.9995, "rmse": ..., "mae": ...},
#  "CD": {"r2": 0.9987, "rmse": ..., "mae": ...}}
```

### 3. Export to ONNX

```python
# MLP surrogates can be exported for deployment
surrogate.export("airfoil_model.onnx")
# Creates: airfoil_model__CL.onnx, airfoil_model__CD.onnx

# Load with onnxruntime
import onnxruntime as ort
session = ort.InferenceSession("airfoil_model__CL.onnx")
cl = session.run(None, {"X": x_new.astype(np.float32)})[0]
```

### 4. Sampling methods

```python
# Latin Hypercube Sampling (default) — space-filling
X, Y = pr.sample(func, bounds=[(0, 1), (-2, 2)], n=100, method="lhs")

# Uniform random
X, Y = pr.sample(func, bounds=[(0, 1), (-2, 2)], n=100, method="uniform")

# Sobol sequences — low-discrepancy, reproducible
X, Y = pr.sample(func, bounds=[(0, 1), (-2, 2)], n=128, method="sobol")

# From existing data
surrogate = pr.Surrogate(params=["alpha", "mach"], outputs=["CL"], method="gp")
surrogate.fit(X, Y)  # X: (n_points, n_params), Y: (n_points, n_outputs)
```

## Built-in analytical functions

All return exact mathematical values — perfect for framework validation with zero CFD.

| Function | Formula | Parameters |
|----------|---------|------------|
| `cl_flat_plate(alpha, camber)` | CL = 2π(α + 2camber) | α: angle of attack [°], camber: ratio |
| `cd_cylinder(reynolds)` | Piecewise Re-dependent CD | Re: Reynolds number |
| `thrust_propeller(rpm, diameter, pitch)` | T = CT·ρ·n²·D⁴ | rpm, diameter [m], pitch [m] |

## Architecture

```
prandtl/
├── __init__.py          # Public API: Surrogate, sample()
├── _surrogate.py        # Core Surrogate class (fit/predict/validate/export)
├── _gaussian.py         # GPyTorch ExactGP wrapper
├── _neural.py           # PyTorch MLP wrapper
├── _sampling.py         # LHS, uniform, Sobol samplers
├── _analytical.py       # Analytical truth functions
└── analytical.py        # Public re-export
```

## Limitations

- **GP ONNX export**: GP models are non-parametric (they need training data for inference) and cannot be exported to ONNX. Use `method='mlp'` if you need exportable surrogates.
- **No multi-fidelity yet**: Single-fidelity only in this release. Multi-fidelity (Co-Kriging) planned.
- **No physics constraints yet**: Pure data-driven fitting. PINN-style physics constraints and Sobolev training planned.
- **CPU only**: CUDA support is available via PyTorch but not yet optimized.

## Roadmap

- [ ] Physics-informed regularization (PDE residuals as loss)
- [ ] Multi-fidelity surrogates (Co-Kriging)
- [ ] Sobolev training (gradient-constrained)
- [ ] Built-in 2D airfoil parameterization
- [ ] OpenFOAM case generation + parsing
- [ ] Isaac Sim force/moment injection plugin

## License

MIT
