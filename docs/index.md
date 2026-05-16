# Prandtl

**CFD surrogate modeling toolkit.** Train fast aerodynamic surrogates — scikit-learn-like API.

```python
import prandtl as pr

# Sample + learn + predict
X, Y = pr.sample(pr.analytical.cl_flat_plate, bounds=[(-5, 15), (0.01, 0.1)], n=100)
surrogate = pr.Surrogate(params=["alpha", "camber"], outputs=["CL"], method="gp")
surrogate.fit(X, Y)
Y_pred = surrogate.predict(X_test)

# Validate
from prandtl import metrics, cross_validate, learning_curve, residual_analysis
```

## The Problem

CFD simulation: **40 minutes** per run.  
You need **100+** parameter combinations → **66 hours**.

Prandtl: Learn from 100 runs → predict the rest in **milliseconds**, error < 0.2%.

## Key Features

<div class="grid cards" markdown>

- **四大后端**

    Gaussian Process (GPyTorch) 适合小数据 + 不确定性。  
    MLP (PyTorch) 适合大规模 + ONNX 导出。  
    Random Forest / Gradient Boosting (scikit-learn) — 无需 PyTorch。

- **不确定性量化**

    `predict_with_uncertainty()` — GP 解析方差，RF 树集成方差。

- **验证套件**

    交叉验证、学习曲线、残差分析 — 量化代理模型是否真正有效。

- **物理约束**

    单调性、凸性、边界值、Sobolev 梯度约束 — 将领域知识直接注入训练。

- **主动学习**

    `ActiveLearner` — 最大标准差/随机采样策略，智能选择下一个采样点。

- **Co-Kriging 多保真**

    `CoKriging` — 结合廉价 + 昂贵仿真数据，构建多保真度代理模型。

- **GPU/CUDA 加速**

    `device='cuda'` 标志 — MLP 后端直接在 GPU 上训练。

- **ONNX 导出**

    将训练好的代理模型部署到任何地方：边缘设备、实时控制环、云端。

- **CFD I/O**

    一行代码解析 OpenFOAM forces 和 SU2 history — 从求解器输出到训练就绪。

</div>

## Quick Example

```python
import prandtl as pr
import numpy as np

# 1. Sample the parameter space
X, Y = pr.sample(
    pr.analytical.cl_flat_plate,
    bounds=[(-5, 15), (0.01, 0.1)],
    n=100, method="lhs", seed=42
)

# 2. Train a surrogate
surrogate = pr.Surrogate(
    params=["alpha", "camber"],
    outputs=["CL"],
    method="gp"
)
surrogate.fit(X, Y)

# 3. Predict on new points
X_test, Y_test = pr.sample(
    pr.analytical.cl_flat_plate,
    bounds=[(-5, 15), (0.01, 0.1)],
    n=30, seed=99
)
Y_pred = surrogate.predict(X_test)

# 4. Evaluate
report = pr.metrics(Y_test, Y_pred)
print(report)  # R² > 0.999 on smooth analytical functions
```

[:octicons-arrow-right-24: Get Started](install.md){ .md-button .md-button--primary }
[:octicons-mark-github-24: GitHub](https://github.com/goodisok/prandtl){ .md-button }
