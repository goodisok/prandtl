# I/O

Read CFD simulation output into training-ready format.

## read_foam_forces

```python
from prandtl import read_foam_forces
X, Y = read_foam_forces("postProcessing/forces/0/coefficient.dat")
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `filepath` | str | Path to OpenFOAM coefficient.dat |

| Returns | Type | Description |
|---------|------|-------------|
| `X` | ndarray (n, p) | Input parameters extracted from file |
| `Y` | ndarray (n, q) | Aerodynamic coefficients (CL, CD, Cm) |

Reads OpenFOAM's `coefficient.dat` (forces function object output). Automatically detects columns:
- Inputs: `alpha`, `AoA`, `mach`, `Re`
- Outputs: `CL`, `CD`, `CMx`, `CMy`, `CMz`

## read_su2_history

```python
from prandtl import read_su2_history
X, Y = read_su2_history("history.csv")
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `filepath` | str | Path to SU2 history CSV |

| Returns | Type | Description |
|---------|------|-------------|
| `X` | ndarray (n, p) | Input parameters |
| `Y` | ndarray (n, q) | Aerodynamic coefficients |

Parses SU2's CSV history output. Same column auto-detection as `read_foam_forces`.
