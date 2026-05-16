# Co-Kriging 多保真

Co-Kriging 结合**低保真**（廉价/近似）和**高保真**（昂贵/精确）数据，构建比单保真 GP 更准确、更高效的多保真度代理模型。

## 动机

| 数据来源 | 成本 | 精度 | 可用数量 |
|----------|------|------|----------|
| 粗网格 CFD / 经验公式 | 低 | 近似 | 多（数百） |
| 细网格 CFD / 风洞实验 | 高 | 精确 | 少（几十） |

Co-Kriging 利用大量廉价数据捕获整体趋势，再用少量昂贵数据修正偏差。

## CoKriging 类

```python
from prandtl import CoKriging

ck = CoKriging(
    params=["alpha", "mach"],
    outputs=["CL"]
)

# 拟合：先廉价数据，后昂贵数据
ck.fit(X_cheap, Y_cheap, X_expensive, Y_expensive)

# 预测
Y_pred = ck.predict(X_test)
```

## 完整示例

```python
import prandtl as pr
from prandtl import CoKriging
import numpy as np

# 低保真数据源：粗网格近似
def cheap_model(alpha, mach):
    # 简化经验公式
    cl = 2 * np.pi * np.radians(alpha) * (1 + 0.1 * mach)
    return {"CL": cl}

# 高保真数据源：精细 CFD
def expensive_model(alpha, mach):
    # 更精确的公式（含非线性修正）
    cl = 2 * np.pi * np.radians(alpha) * (1 + 0.12 * mach - 0.03 * mach**2)
    return {"CL": cl}

# 大量低保真采样
X_cheap, Y_cheap = pr.sample(
    cheap_model,
    bounds=[(-5, 15), (0.15, 0.85)],
    n=200, method="lhs", seed=42
)

# 少量高保真采样
X_expensive, Y_expensive = pr.sample(
    expensive_model,
    bounds=[(-5, 15), (0.15, 0.85)],
    n=30, method="lhs", seed=42
)

# 构建 Co-Kriging 模型
ck = CoKriging(
    params=["alpha", "mach"],
    outputs=["CL"]
)
ck.fit(X_cheap, Y_cheap, X_expensive, Y_expensive)

# 预测
X_test, Y_test = pr.sample(
    expensive_model,
    bounds=[(-5, 15), (0.15, 0.85)],
    n=50, seed=99
)
Y_pred = ck.predict(X_test)

# 评估
from prandtl import metrics
report = metrics(Y_test, Y_pred)
print(f"R²: {report['CL']['r2']:.4f}")
```

## 关键要点

- **两阶段拟合**：先对廉价数据拟合 GP（趋势），再对昂贵数据拟合差分 GP（修正）。
- **优势**：用 30 个昂贵点 + 200 个廉价点，通常优于仅用 30 个昂贵点的单保真 GP。
- **限制**：当前版本仅支持两个保真度级别。多级 Co-Kriging（3+）在开发路线图中。
