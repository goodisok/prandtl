# Sobolev 约束

Sobolev 训练将**梯度匹配**作为训练约束，使代理模型不仅拟合输出值，还拟合输出对输入的偏导数 — 适用于已知物理梯度信息的场景。

## GradientConstraint

```python
from prandtl import GradientConstraint

constraint = GradientConstraint(
    param_idx=0,           # 对第 0 个参数求梯度
    grad_values=grad_data, # 梯度真值 (n_points,)
    weight=0.1             # 约束权重
)
```

## 完整示例

```python
import prandtl as pr
from prandtl import GradientConstraint
import numpy as np

# 生成训练数据
X, Y = pr.sample(
    pr.analytical.cl_flat_plate,
    bounds=[(-5, 15), (0.01, 0.1)],
    n=100, method="lhs", seed=42
)

# 计算梯度真值：CL 对 alpha 的导数
# d(CL)/d(alpha) = d(2π(α + 2camber))/d(alpha) = 2π (常数)
alpha_grad = 2 * np.pi * np.ones((100, 1))

# 构建 Sobolev 约束
grad_constraint = GradientConstraint(
    param_idx=0,
    grad_values=alpha_grad,
    weight=0.1
)

# 训练（仅 MLP 支持）
surrogate = pr.Surrogate(
    params=["alpha", "camber"],
    outputs=["CL"],
    method="mlp"
)
surrogate.fit(
    X, Y,
    physics=[grad_constraint],
    n_iter=3000,
    lr=0.001
)

# 验证：模型不仅准确预测 CL，导数行为也符合物理
Y_pred = surrogate.predict(X_test)
```

## 参数说明

| 参数 | 说明 |
|------|------|
| `param_idx` | 对哪个输入参数求偏导（从 0 开始） |
| `grad_values` | 梯度目标值，形状 `(n_points,)` 或 `(n_points, 1)` |
| `weight` | 梯度约束的权重（0.01–1.0）。权重越大，强制匹配越严格 |

## 应用场景

- **已知解析梯度**：从理论/经验公式推导出导数关系
- **伴随求解器**：CFD 伴随求解器可同时输出函数值和梯度
- **物理一致性**：强制代理模型的导数行为与物理定律一致

!!! warning "仅限 MLP"
    `GradientConstraint` 基于自动微分，仅支持 `method="mlp"`。GP 和树模型后端不支持此约束。

## 与其他约束组合

```python
from prandtl import Monotonicity, BoundaryValue, GradientConstraint

constraints = [
    Monotonicity(param_idx=0, sign=1, weight=0.1),
    BoundaryValue({"alpha": 0.0}, {"CL": 0.0}, weight=10.0),
    GradientConstraint(param_idx=0, grad_values=alpha_grad, weight=0.05),
]

surrogate.fit(X, Y, physics=constraints, n_iter=5000, lr=0.001)
```
