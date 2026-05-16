# Analytical Functions

Built-in truth functions for pipeline validation — zero CFD required.

## cl_flat_plate

```python
from prandtl.analytical import cl_flat_plate
Y = cl_flat_plate({"alpha": 5.0, "camber": 0.04})
# → {"CL": 0.632}
```

Thin airfoil theory: CL = 2π(α + 2camber), α in degrees.

| Parameter | Unit | Range |
|-----------|------|-------|
| `alpha` | degrees | [-20, 20] |
| `camber` | fraction | [0, 0.15] |

## cd_cylinder

```python
from prandtl.analytical import cd_cylinder
Y = cd_cylinder({"Re": 1000.0})
# → {"CD": 2.0}
```

Laminar cylinder drag: CD = 24/Re + 6/(1+√Re) + 0.4

| Parameter | Unit | Range |
|-----------|------|-------|
| `Re` | — | [0.1, 1e5] |

## thrust_propeller

```python
from prandtl.analytical import thrust_propeller
Y = thrust_propeller({"rpm": 5000.0, "pitch": 0.3, "diameter": 0.5})
# → {"T": 12.5}
```

Momentum theory thrust: T = ρ·n²·D⁴·C_T(pitch/D)
