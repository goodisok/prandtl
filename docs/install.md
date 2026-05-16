# Installation

## pip

```bash
pip install prandtl-cfd             # base (numpy, scipy, torch)
pip install prandtl-cfd[gp]         # Gaussian Process backend (GPyTorch)
pip install prandtl-cfd[export]     # ONNX export support
pip install prandtl-cfd[all]        # everything (gp + export)
```

## Extras

| Extra | What it installs | When you need it |
|-------|-----------------|------------------|
| *none* | numpy, scipy, torch | Sampling + MLP training + metrics |
| `[gp]` | GPyTorch | Gaussian Process surrogates |
| `[export]` | onnx, onnxruntime | Deploying MLP surrogates |
| `[all]` | gp + export | Everything |

## Check installation

```python
import prandtl as pr
print(pr.__version__)  # 0.3.0

# Verify GP backend
X, Y = pr.sample(pr.analytical.cl_flat_plate, bounds=[(0, 5), (0.01, 0.05)], n=20)
s = pr.Surrogate(params=["alpha", "camber"], outputs=["CL"], method="gp")
s.fit(X, Y)
print(s.validate(X, Y))  # should show R² ≈ 1.0
```
