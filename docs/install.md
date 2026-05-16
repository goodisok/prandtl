# Installation

## pip

```bash
pip install prandtl-cfd             # base (numpy, scipy, torch)
pip install prandtl-cfd[gp]         # Gaussian Process backend (GPyTorch)
pip install prandtl-cfd[export]     # ONNX export support
pip install prandtl-cfd[tree]       # 树模型后端 (Random Forest / Gradient Boosting)
pip install prandtl-cfd[all]        # 全部功能 (gp + export + tree)
```

## Extras

| Extra | What it installs | When you need it |
|-------|-----------------|------------------|
| *none* | numpy, scipy, torch | Sampling + MLP training + metrics |
| `[gp]` | GPyTorch | Gaussian Process surrogates |
| `[export]` | onnx, onnxruntime | 部署 MLP 代理模型 |
| `[tree]` | scikit-learn | Random Forest / Gradient Boosting 后端 |
| `[all]` | gp + export + tree | 全部功能 |

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
