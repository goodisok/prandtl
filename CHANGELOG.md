# Changelog

## [0.5.0] — 2026-06-01

### Added
- **GPU/CUDA acceleration** — `Surrogate(device='cuda')` opt-in flag; CPU stays default
- **Uncertainty quantification API** — `predict_with_uncertainty(X)` returns `(mean, std)` for GP and RF backends
- **New model backends**: Random Forest (`method='rf'`) and Gradient Boosting (`method='gb'`)
  - RF provides tree-variance uncertainty (standard deviation across ensemble)
  - GB supports quantile regression via standalone `GradientBoosting.fit_with_uncertainty()`
- **Active learning** — `ActiveLearner` Bayesian optimization with batch selection strategies
- **Multi-fidelity modeling** — `CoKriging` two-level Gaussian process for combining cheap and expensive data
- **Sobolev training** — `GradientConstraint` for physics-informed gradient matching in MLP
- **Analytical benchmark functions** — `NACA0012`, `RAE2822`, full suite in `prandtl.analytical`
- Standalone tree constructors: `prandtl.RandomForest(...)` and `prandtl.GradientBoosting(...)`
- 87 tests (up from 82)

### Changed
- RF/GB models save under `_tree.py`; `_surrogate.py` dispatches to them via `method='rf'/'gb'`
- `predict_with_uncertainty()` now raises `NotImplementedError` for GB (points user to quantile regression)
- `export()` message generalized from GP-only to all non-MLP backends

## [0.4.0] — 2026-05-31

### Fixed
- **GP kernel validation**: Unknown `gp_kernel` names now raise `ValueError` instead of silently falling back to RBF
- **Sample count validation**: `fit()` now checks `X.shape[0] == Y.shape[0]` before training
- **BoundaryValue weight**: `BoundaryValue.weight` was ignored during MLP training — now correctly applied
- **Sobol sampling**: second param now explicitly positional (removed unreliable inference)

### Added
- **Sobol sampling** (`sample_method='sobol'`) via scipy's Sobol' sequence
- **Matern kernel variants**: `matern15` (ν=1.5), `matern25` (ν=0.5), `matern52` (ν=2.5) for GP surrogate
- **Analytical test suite**: 5 tests for flat plate, cylinder, and propeller benchmark functions
- **I/O error-path tests**: non-numeric data, empty files, directory paths
- 82 tests total (up from 62)
- CI pipeline (`.github/workflows/test.yml`) — pytest + ruff on Python 3.11–3.13
- `CONTRIBUTING.md`

## [0.3.0] — 2026-05-17

### Added
- **Validation module** (`_validate.py`): four data-independent validation tools
  - `metrics(Y_true, Y_pred)` — 7 regression metrics (R², RMSE, MAE, MAPE, max_error, max_relative_error, explained_variance)
  - `residual_analysis(Y_true, Y_pred)` — Shapiro-Wilk normality test, skewness, kurtosis, max residual index
  - `cross_validate(surrogate, X, Y, cv=N)` — k-fold CV with per-output aggregation (mean ± std across folds)
  - `learning_curve(surrogate, X, Y, sizes=[...])` — performance vs training set size curve
- Public API: `cross_validate`, `learning_curve`, `metrics`, `residual_analysis`
- 22 new tests (`tests/test_validate.py`)
- `onnxscript` added to `[export]` extras (required by PyTorch 2.x ONNX export)

### Changed
- `cross_validate` computes per-fold MAE directly from predictions (not RMSE approximation)

## [0.2.0] — 2026-05-17

### Added
- **Physics-informed constraints** (`_physics.py`): Monotonicity, Convexity, BoundaryValue, CustomConstraint — plug into MLP training as soft regularization
- **CFD data I/O** (`_io.py`): `read_foam_forces()` and `read_su2_history()` — parse simulation output into `(X, Y)` format
- Public API exports: `Monotonicity`, `Convexity`, `BoundaryValue`, `CustomConstraint`, `read_foam_forces`, `read_su2_history`
- 13 new tests (physics + IO)

### Changed
- `torch` moved to base dependencies (was optional)
- Simplified extras: removed `[mlp]`, `[all]` now = `[gp,export]`
- `Surrogate.fit()` accepts `physics=` parameter
- `BoundaryValue` accepts both `dict` and `numpy.ndarray` input

### Fixed
- PyPI metadata: classifiers, homepage URLs
- TOML parser error when `dependencies` followed `[project.urls]`

## [0.1.0] — 2026-05-16

### Added
- `prandtl.Surrogate` with GP (GPyTorch) and MLP (PyTorch) backends
- `prandtl.sample()` — LHS, uniform, Sobol sampling
- `prandtl.analytical` — 3 built-in validation functions
- ONNX export (MLP only)
- 23 tests, bilingual README
