# Changelog

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
