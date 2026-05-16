# Prandtl

[English](#prandtl) | [中文](README_zh.md)

CFD surrogate modeling toolkit. Train fast surrogates for aerodynamic predictions — scikit-learn-like API.

📖 **[Full Documentation](https://goodisok.github.io/prandtl/)** — install guide, user guide, API reference, examples

```python
import prandtl as pr

# Sample parameter space + analytical truth
X, Y = pr.sample(pr.analytical.cl_flat_plate, bounds=[(-5, 15), (0.01, 0.1)], n=100)

# Train Gaussian Process surrogate
surrogate = pr.Surrogate(params=["alpha", "camber"], outputs=["CL"], method="gp")
surrogate.fit(X, Y)

# Predict + validate
Y_pred = surrogate.predict(X_test)
report = pr.metrics(Y_test, Y_pred)
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
pip install prandtl-cfd             # Base (numpy, scipy, torch)
pip install prandtl-cfd[gp]         # Gaussian Process backend (GPyTorch)
pip install prandtl-cfd[export]     # ONNX export support
pip install prandtl-cfd[all]        # Everything (gp + export)
```

## v0.3.0 highlights

### Cross-validation & metrics (new)

```python
# K-fold cross-validation — one line
scores = pr.cross_validate(surrogate, X, Y, cv=5)
# → {"CL": {"mae_mean": 0.012, "mae_std": 0.003, "r2_mean": 0.999, ...}}

# Extended metrics beyond RMSE/R²
metrics = pr.metrics(Y, Y_pred)
# → {"CL": {"r2": 0.9996, "rmse": 0.0010, "mae": 0.0008,
#            "max_re": 0.0034, "explained_variance": 0.9996}}

# Residual diagnostics
res = pr.residual_analysis(Y, Y_pred)
# → {"CL": {"shapiro_stat": 0.987, "shapiro_p": 0.42,  # p>0.05 → normal ✓
#            "skewness": -0.15, "kurtosis": 2.91, "max_residual_idx": 7,
#            "residuals": array([...])}}

# Learning curve — performance vs training size
curve = pr.learning_curve(surrogate, X, Y, sizes=[20, 40, 60, 80, 100])
# → {"train_sizes": [20, 40, 60, 80, 100],
#     "train_mae": [0.005, 0.008, 0.010, 0.011, 0.012],
#     "val_mae":   [0.018, 0.014, 0.013, 0.012, 0.012]}
```

### Physics constraints (v0.2.0+)

```python
from prandtl import Monotonicity, Convexity, BoundaryValue

surrogate.fit(X, Y, physics=[
    Monotonicity(param_idx=0, sign=1, weight=0.1),          # CL ↑ monotonically with α
    BoundaryValue({"alpha": 0.0}, {"CL": 0.0}, weight=10.0), # CL=0 at α=0
    Convexity(param_idx=0, sign=-1, weight=0.05),            # concave drag polar
], n_iter=500, lr=0.01)
```

### CFD data I/O

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
| **Validation suite** | Cross-validation, learning curves, residual analysis, and extended metrics (R², RMSE, MAE, MaxRE, Explained Variance) |
| **Physics constraints** | Monotonicity, convexity, and boundary value soft constraints during MLP training |
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
Y_pred = surrogate.predict(X_test)
report = pr.metrics(Y_test, Y_pred)
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
Y_pred = surrogate.predict(X_test)
report = pr.metrics(Y_test, Y_pred)
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

### 5. Physics-informed training

```python
from prandtl import Monotonicity, BoundaryValue, Convexity

constraints = [
    Monotonicity(param_idx=0, sign=1, weight=0.1),
    # CL must increase with alpha (param_idx=0). sign=+1 enforces monotonic increase.
    BoundaryValue({"alpha": 0.0}, {"CL": 0.0}, weight=10.0),
    # At alpha=0°, CL must be 0. High weight = strict constraint.
    Convexity(param_idx=0, sign=-1, weight=0.05),
    # Concave relationship (sign=-1) — e.g., drag polar curvature.
]

surrogate = pr.Surrogate(params=["alpha", "mach"], outputs=["CL", "CD"], method="mlp")
surrogate.fit(X, Y, physics=constraints, n_iter=500, lr=0.01)
```

### 6. Cross-validation

```python
# 5-fold CV: train on 80%, test on 20%, repeat 5 times
scores = pr.cross_validate(surrogate, X, Y, cv=5, verbose=True)
print(f"MAE: {scores['CL']['mae_mean']:.4f} ± {scores['CL']['mae_std']:.4f}")
print(f"R²:  {scores['CL']['r2_mean']:.4f} ± {scores['CL']['r2_std']:.4f}")

# All outputs scored automatically
# {'CL': {'mae_mean': ..., 'mae_std': ..., 'rmse_mean': ..., 'r2_mean': ..., ...},
#  'CD': {'mae_mean': ..., ...}}
```

### 7. Learning curve

```python
# See how performance scales with training data
curve = pr.learning_curve(surrogate, X, Y, sizes=[10, 20, 50, 100, 150])

# Interpret: if val_mae plateaus, you have enough data.
# If train_mae ≪ val_mae, you're overfitting — try simpler model or fewer iterations.
print(f"Final train MAE: {curve['train_mae'][-1]:.4f}")
print(f"Final val MAE:   {curve['val_mae'][-1]:.4f}")
```

### 8. Residual analysis

```python
res = pr.residual_analysis(Y_test, Y_pred)

# Shapiro-Wilk normality test: p > 0.05 → residuals are normally distributed ✓
for output in res:
    r = res[output]
    print(f"{output}:")
    print(f"  Shapiro-Wilk p={r['shapiro_p']:.3f} {'✓' if r['shapiro_p'] > 0.05 else '✗'}")
    print(f"  Skewness={r['skewness']:.3f}, Kurtosis={r['kurtosis']:.3f}")
    print(f"  Max residual at index {r['max_residual_idx']}")

# High skewness → systematic bias. High kurtosis → outliers.
# Non-normal residuals → model is missing physics or needs more data.
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
├── __init__.py          # Public API: Surrogate, sample(), cross_validate(), metrics(), ...
├── _surrogate.py        # Core Surrogate class (fit/predict/validate/export)
├── _gaussian.py         # GPyTorch ExactGP wrapper
├── _neural.py           # PyTorch MLP wrapper
├── _validate.py         # Cross-validation, learning curves, residual analysis, metrics
├── _physics.py          # Physics-informed constraints (Monotonicity, Convexity, BoundaryValue)
├── _sampling.py         # LHS, uniform, Sobol samplers
├── _io.py               # CFD data I/O (OpenFOAM forces, SU2 history)
├── _analytical.py       # Analytical truth functions
└── analytical.py        # Public re-export
```

## Limitations

- **GP ONNX export**: GP models are non-parametric (they need training data for inference) and cannot be exported to ONNX. Use `method='mlp'` if you need exportable surrogates.
- **No multi-fidelity yet**: Single-fidelity only in this release. Multi-fidelity (Co-Kriging) planned.
- **CPU only**: CUDA is available via PyTorch but not yet integrated. On near-term roadmap.

## Roadmap

**Done:**
- [x] GP + MLP dual backends
- [x] Physics-informed constraints (Monotonicity, Convexity, BoundaryValue)
- [x] Validation suite (cross-validation, learning curves, residual analysis)
- [x] CFD data I/O (OpenFOAM, SU2)
- [x] ONNX export (MLP)

**Near-term (v0.4–v0.5):**
- [ ] GPU/CUDA support — PyTorch backend already CUDA-capable; needs opt-in flag
- [ ] Uncertainty quantification API — GP `.predict()` returns predictive variance
- [ ] Active learning / Bayesian optimization — "where to sample next?"
- [ ] More analytical benchmark functions (NACA 0012, RAE 2822, etc.)

**Mid-term (v0.6+):**
- [ ] Multi-fidelity surrogates (Co-Kriging)
- [ ] Sobolev training (gradient-constrained fitting)
- [ ] Additional model backends (Random Forest, Gradient Boosting)

## License

MIT
