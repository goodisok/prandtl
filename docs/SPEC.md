# Spec: Prandtl — CFD Surrogate Modeling Toolkit

## Objective

**What**: A Python toolkit that lets simulation engineers train, validate, and export surrogate models for CFD with minimal code. Three lines: sample → fit → export.

**Who**: Simulation engineers who need fast aerodynamic predictions without running full CFD every time.

**Why**: Existing solutions (scikit-learn GP, SMT) are generic ML tools. None provide the domain-specific workflow of parameter sampling → surrogate training → validation reports → simulator-ready export in a single package. Every aerospace/robotics company builds this internally; no open-source standard exists.

**Success**: An engineer can replace a CFD simulation loop with a trained surrogate model that predicts CL/CD/CM at sub-millisecond latency with >95% R² on held-out data.

## Tech Stack

| Component | Choice | Version |
|-----------|--------|---------|
| Language | Python | ≥ 3.10 |
| Gaussian Process | GPyTorch | latest |
| Neural Network | PyTorch | latest |
| Export | ONNX + onnxruntime | latest |
| Sampling | scipy (LHS) | latest |
| Math | numpy | latest |
| Build | setuptools / pyproject.toml | PEP 621 |
| Test | pytest | latest |
| Lint | ruff | latest |

## Commands

```
Install:   pip install -e .
Test:      pytest tests/ -v
Lint:      ruff check src/
Format:    ruff format src/
Typecheck: mypy src/
```

## Project Structure

```
prandtl/
├── pyproject.toml          # PEP 621 build config
├── README.md
├── docs/
│   └── SPEC.md             # This file
├── src/
│   └── prandtl/
│       ├── __init__.py     # Public API surface
│       ├── _sampling.py    # LHS, uniform, Sobol samplers
│       ├── _analytical.py  # Analytical truth functions for validation
│       ├── _surrogate.py   # Core Surrogate class (unified interface)
│       ├── _gaussian.py    # GP backend (GPyTorch)
│       ├── _neural.py      # MLP backend (PyTorch)
│       ├── _validate.py    # Cross-validation, residual analysis, metrics
│       └── _export.py      # ONNX export + simulator interface stubs
└── tests/
    ├── test_sampling.py
    ├── test_analytical.py
    ├── test_surrogate.py
    ├── test_gaussian.py
    ├── test_neural.py
    ├── test_validate.py
    └── test_export.py
```

Underscore-prefixed private modules. Only `__init__.py` exposes the public API.

## Code Style

```python
"""One-line module docstring."""

from typing import Optional

import numpy as np
import torch


class Surrogate:
    """CFD surrogate model with a scikit-learn-like interface.

    Parameters
    ----------
    params : list of str
        Names of input parameters, e.g. ['alpha', 'mach', 'camber'].
    outputs : list of str
        Names of output quantities, e.g. ['CL', 'CD'].
    method : str
        Backend: 'gp' (Gaussian Process) or 'mlp' (neural network).
    """

    def __init__(
        self,
        params: list[str],
        outputs: list[str],
        method: str = "gp",
    ) -> None:
        ...

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        n_iter: int = 100,
        verbose: bool = True,
    ) -> "Surrogate":
        """Train the surrogate on (X, Y) data. Returns self for chaining."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted outputs for given inputs."""
        ...

    def validate(self, X_test: np.ndarray, Y_test: np.ndarray) -> dict:
        """Return dict with R², RMSE, max_error per output."""
        ...
```

Key conventions:
- Google-style docstrings (numpy docstring format for scientific audience)
- Type hints on all public methods
- Private modules prefixed with `_`
- Classes: PascalCase. Functions/variables: snake_case
- 100 char line limit
- Explicit `*` for keyword-only arguments where appropriate

## Testing Strategy

- **Framework**: pytest with `--strict-markers`
- **Location**: `tests/` mirrors `src/prandtl/`
- **Coverage target**: >90% on core modules
- **Test levels**:
  - Unit: each module in isolation with small synthetic data
  - Integration: `Surrogate.fit() → .predict() → .validate()` end-to-end with analytical truth
- **No GPU required** — all tests runnable on CPU with small synthetic data
- **CI** (future): GitHub Actions on push, CPU only

## API Design (MVP)

### The Three-Line Interface

```python
import prandtl as pr

# 1. Sample
X, Y = pr.sample(pr.analytical.cl_flat_plate, bounds=[(-5, 15), (0.01, 0.1)], n=100)

# 2. Fit
surrogate = pr.Surrogate(params=["alpha", "camber"], outputs=["CL"]).fit(X, Y)

# 3. Validate
report = surrogate.validate(*pr.sample(pr.analytical.cl_flat_plate, bounds=[...], n=20))
print(f"R² = {report['CL']['r2']:.4f}")  # → R² = 0.9998
```

### Module Breakdown

#### `prandtl.sample(func, bounds, n, method='lhs')`
Sample parameter space and evaluate truth function.
- `func`: Callable that takes `**params` and returns dict of outputs
- `bounds`: List of (low, high) tuples per parameter
- `n`: Number of design points
- `method`: 'lhs' | 'uniform' | 'sobol'
- Returns: `(X: np.ndarray, Y: np.ndarray)`

#### `prandtl.Surrogate(params, outputs, method='gp')`
Main class. Unified interface over GP and MLP backends.

#### `analytical` module
Built-in truth functions for framework validation:
- `cl_flat_plate(alpha, camber)` → CL = 2π(α + 2c)  [thin airfoil theory]
- `cd_cylinder(reynolds)` → empirical drag curve
- `thrust_propeller(rpm, diameter, pitch)` → T = CT·ρ·n²·D⁴

## Success Criteria (MVP)

- [x] `pip install -e .` succeeds
- [x] `pr.sample()` returns correct shapes with LHS and uniform methods
- [x] GP surrogate fits `cl_flat_plate` with R² > 0.99 on 100 train / 20 test points
- [x] MLP surrogate fits `cl_flat_plate` with R² > 0.99 on 100 train / 20 test points
- [x] `surrogate.validate()` returns dict with r2, rmse, max_error per output
- [x] `surrogate.export('model.onnx')` produces valid ONNX file
- [x] All tests pass: `pytest tests/ -v`
- [x] Ruff lint clean: `ruff check src/`

## Boundaries

**Always do:**
- Run tests before declaring a feature done
- Write docstrings on all public functions and classes
- Use type hints on all public API
- Keep imports minimal — no unused deps

**Ask first:**
- Adding new dependencies beyond numpy, scipy, torch, gpytorch, onnx
- Changing the public API (anything in `__init__.py`)
- Adding GPU-dependent code paths

**Never do:**
- Hard-code file paths
- Assume internet access at runtime
- Import heavy deps at module level (lazy import in `__init__.py`)

## Open Questions

None — all addressed in assumptions above.
