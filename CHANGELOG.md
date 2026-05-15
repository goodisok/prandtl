# Changelog

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
