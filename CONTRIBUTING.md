# Contributing to Prandtl

Thanks for your interest in contributing!

## Quick Start

```bash
git clone https://github.com/goodisok/prandtl.git
cd prandtl
pip install -e ".[torch]"
pip install pytest
python -m pytest tests/ -v
```

## Development Workflow

1. **Fork & branch** — create a feature branch from `main`
2. **Write tests first** — follow TDD, all changes need test coverage
3. **Run full test suite** — `python -m pytest tests/` must pass
4. **Open PR** — describe what and why, link related issues

## Code Style

- Python 3.11+ with type annotations
- Follow `ruff` linting rules (`ruff check` must be clean)
- Use NumPy docstring style
- Keep modules focused — one responsibility per module

## Testing

- 82+ tests covering surrogate modeling, I/O, physics constraints, and validation
- Run: `python -m pytest tests/ -v`
- Coverage: `python -m pytest tests/ --cov=prandtl`

## Project Structure

```
src/prandtl/
  __init__.py      # Public API
  _sampling.py     # LHS, Sobol, uniform sampling
  _io.py           # OpenFOAM/SU2 file parsers
  _physics.py      # Monotonicity, Convexity, BoundaryValue constraints
  _gaussian.py     # Exact GP with RBF/Matern kernels
  _neural.py       # MLP with physics-informed loss
  _surrogate.py    # Surrogate class (orchestrator)
  _analytical.py   # Benchmark functions (flat plate, cylinder, propeller)
  _validate.py     # Cross-validation, learning curves, residual analysis
tests/
  test_e2e.py      # End-to-end surrogate tests
  test_io.py       # File I/O tests
  test_physics.py  # Physics constraint tests
  test_validate.py # Validation module tests
```

## Questions?

Open an issue on GitHub or submit a PR.
