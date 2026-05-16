# Sobolev 约束

Sobolev 训练将**梯度匹配**作为训练约束，使 GP 代理模型不仅拟合输出值，还拟合输出对输入的偏导数 — 适用于已知物理梯度信息的场景（如 CFD 伴随求解器）。

## soboloev 类

```python
from prandtl import soboloev

mdl = soboloev(
    params=["alpha", "mach"],
    output="CL",
    kernel="rbf",        # "rbf" 或 "matern52"
    grad_weight=0.5      # 梯度约束权重 (0~1, 默认 0.5)
)
```

| 参数 | 说明 |
|------|------|
| `params` | 输入参数名列表 |
| `output` | 输出变量名 |
| `kernel` | GP 核函数：`"rbf"`（默认）或 `"matern52"` |
| `grad_weight` | 梯度匹配在联合损失中的权重。0 = 标准 GP，1 = 纯梯度匹配 |

## 完整示例

```python
import prandtl as pr
import numpy as np

# 1. 已知梯度的真实函数
def f(x):
    """f(x) = sin(3x)，梯度 f'(x) = 3cos(3x)"""
    return np.sin(3*x), 3*np.cos(3*x)

# 2. 生成训练数据（值 + 梯度）
X = np.random.uniform(0, 2, (8, 1))
Y, dY = f(X)

# 3. 训练 Sobolev GP
mdl = pr.soboloev(params=["x"], output="y", kernel="rbf", grad_weight=0.5)
mdl.fit(X, Y, dY)

# 4. 预测输出值
y_pred, y_std = mdl.predict(np.array([[1.0]]))

# 5. 预测梯度（解析计算，非有限差分）
grad_pred = mdl.predict_gradient(np.array([[1.0]]))

print(f"预测值: {y_pred[0, 0]:.4f} ± {y_std[0, 0]:.4f}")
print(f"预测梯度: {grad_pred[0, 0]:.4f}")
print(f"真实梯度: {3*np.cos(3*1.0):.4f}")
```

## 方法一览

| 方法 | 说明 |
|------|------|
| `fit(X, Y, dY)` | 用函数值和梯度数据训练 |
| `predict(X)` | 返回 `(mean, std)` |
| `predict_gradient(X)` | 解析计算后验均值梯度，无需有限差分 |

## 应用场景

| 场景 | 梯度来源 |
|------|----------|
| CFD 伴随求解器 | 计算 ∂CL/∂α 和 ∂CL/∂Mach 几乎零成本 |
| 已知解析函数 | 手动求导 |
| 有限差分（备选） | 当伴随不可用时 |

!!! warning "仅限 GP"
    Sobolev 训练利用 GP 核函数的解析导数。MLP 和其他后端不支持此功能。

!!! tip "grad_weight 调节"
    梯度数据有噪声时降低 `grad_weight`（如 0.1~0.3）。梯度质量高时可提高（0.7~0.9）以获得更好的导数精度。
