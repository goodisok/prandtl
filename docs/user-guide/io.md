# CFD Data I/O

Read CFD simulation output directly into training-ready format.

## OpenFOAM Forces

```python
from prandtl import read_foam_forces

X, Y = read_foam_forces("postProcessing/forces/0/coefficient.dat")
# X: (n_time_steps, n_params) — rows for each time step
# Y: (n_time_steps, n_outputs) — CL, CD, Cm columns
surrogate.fit(X, Y)
```

Reads OpenFOAM's `coefficient.dat` (forces function object output). Automatically parses the header to identify columns.

## SU2 History

```python
from prandtl import read_su2_history

X, Y = read_su2_history("history.csv")
# Parses SU2's CSV history output
```

## Column Detection

Both functions auto-detect aerodynamic coefficients:
- `CL`, `CD`, `CMx`, `CMy`, `CMz` → outputs
- `alpha`, `AoA`, `mach`, `Re` → inputs
