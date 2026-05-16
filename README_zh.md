# Prandtl · 普朗特

CFD 代理模型工具包。几行代码训练气动预测的快速代理模型——API 风格和 scikit-learn 一样直觉。

[English](README.md) | **中文**

📖 **[完整文档](https://prandtl.pages.dev/)** — 安装指南、用户手册、API 参考、示例

```python
import prandtl as pr

# 参数空间采样 + 解析真值（零 CFD 依赖）
X, Y = pr.sample(pr.analytical.cl_flat_plate, bounds=[(-5, 15), (0.01, 0.1)], n=100)

# 训练高斯过程代理模型
surrogate = pr.Surrogate(params=["alpha", "camber"], outputs=["CL"], method="gp")
surrogate.fit(X, Y)

# 预测 + 验证
Y_pred = surrogate.predict(X_test)
report = pr.metrics(Y_test, Y_pred)
print(report)  # {"CL": {"r2": 0.9998, "rmse": 0.0012, "mae": 0.0010}}

# 导出部署
surrogate.export("model.onnx")  # 每个输出一个 .onnx 文件
```

## 这有什么用？

**问题**：你的无人机旋翼升力有多大？ → 跑一次 CFD 仿真，40 分钟。

**你其实想问 100 个不同转速-攻角组合** → 那是 40×100 = 66 小时。

Prandtl 的做法：从 100 个测点学出规律 → 其他 10,000 个组合**毫秒级预测**，误差 < 0.2%。

用大白话说：CFD 仿真相当于一个**昂贵的计算器**——每按一次花半小时。Prandtl 做的事是**克隆这个计算器**——克隆品秒出结果，跟原件几乎一样准。

换个角度：这是「机器学习」最务实的用法——不是生成图片、不是聊天、不是推荐——就是**学一个函数，替代另一个太慢的函数**。

## 安装

```bash
pip install prandtl-cfd             # 基础版（numpy, scipy, torch）
pip install prandtl-cfd[gp]         # 高斯过程后端（GPyTorch）
pip install prandtl-cfd[export]     # ONNX 导出支持
pip install prandtl-cfd[all]        # 全部安装
```

## v0.5.0 亮点

### 新模型后端：随机森林 & 梯度提升

```python
# 随机森林 — 无需 PyTorch/GPyTorch
surrogate = pr.Surrogate(params=["alpha", "mach"], outputs=["CL", "CD"], method="rf")
surrogate.fit(X, Y)
Y_pred = surrogate.predict(X_test)

# RF 不确定度：树集成标准差
Y_mu, Y_std = surrogate.predict_with_uncertainty(X_test)
```

### 主动学习 — "下一步该在哪里采样？"

```python
from prandtl import ActiveLearner

learner = ActiveLearner(surrogate, X_pool, strategy="max_std")
X_next = learner.query(n=10)          # 选取 10 个最不确定的点
surrogate.fit(X_next, Y_new)          # 标注并重新训练
```

### Co-Kriging：多保真代理模型

```python
from prandtl import CoKriging

ck = CoKriging(params=["alpha"], outputs=["CL"])
ck.fit(X_cheap, Y_cheap, X_expensive, Y_expensive)
Y_pred = ck.predict(X_test)
```

### GPU/CUDA 支持

```python
surrogate = pr.Surrogate(params=["alpha"], outputs=["CL"], method="mlp", device="cuda")
surrogate.fit(X, Y)  # 在 GPU 上训练
```

### 更多

- **Sobolev 训练** — `GradientConstraint` 用于物理信息梯度匹配
- **不确定度量化** — GP 和 RF 支持 `predict_with_uncertainty()`
- **解析基准函数** — `prandtl.analytical` 新增 `NACA0012`、`RAE2822`

## v0.3.0 新功能

### 交叉验证与评估指标（新增）

```python
# K 折交叉验证——一行搞定
scores = pr.cross_validate(surrogate, X, Y, cv=5)
# → {"CL": {"mae_mean": 0.012, "mae_std": 0.003, "r2_mean": 0.999, ...}}

# 扩展指标——不止 RMSE 和 R²
metrics = pr.metrics(Y, Y_pred)
# → {"CL": {"r2": 0.9996, "rmse": 0.0010, "mae": 0.0008,
#            "max_re": 0.0034, "explained_variance": 0.9996}}

# 残差诊断——检查模型是否有系统偏差
res = pr.residual_analysis(Y, Y_pred)
# → {"CL": {"shapiro_stat": 0.987, "shapiro_p": 0.42,  # p>0.05 → 正态 ✓
#            "skewness": -0.15, "kurtosis": 2.91, "max_residual_idx": 7,
#            "residuals": array([...])}}

# 学习曲线——数据量够了吗？
curve = pr.learning_curve(surrogate, X, Y, sizes=[20, 40, 60, 80, 100])
# → {"train_sizes": [20, 40, 60, 80, 100],
#     "train_mae": [0.005, 0.008, 0.010, 0.011, 0.012],
#     "val_mae":   [0.018, 0.014, 0.013, 0.012, 0.012]}
```

### 物理约束（v0.2.0 起）

```python
from prandtl import Monotonicity, Convexity, BoundaryValue

surrogate.fit(X, Y, physics=[
    Monotonicity(param_idx=0, sign=1, weight=0.1),           # CL 随 α 单调增
    BoundaryValue({"alpha": 0.0}, {"CL": 0.0}, weight=10.0),  # α=0 时 CL=0
    Convexity(param_idx=0, sign=-1, weight=0.05),             # 阻力极线凹性
], n_iter=500, lr=0.01)
```

### CFD 数据读写

```python
from prandtl import read_foam_forces, read_su2_history

X, Y = read_foam_forces("postProcessing/forces/0/coefficient.dat")
# → 直接喂给 surrogate.fit(X, Y)
```

## 能做什么

Prandtl 让你用快速 ML 代理模型替代昂贵的 CFD 仿真——无需写任何 ML 样板代码。

| 特性 | 说明 |
|---------|------------|
| **四种后端** | 高斯过程（`gp`）、MLP（`mlp`）、随机森林（`rf`）、梯度提升（`gb`）——树模型无需 PyTorch |
| **不确定度** | `predict_with_uncertainty()`——GP 解析方差、RF 树集成方差 |
| **零 CFD 依赖** | 内置解析真值函数（薄翼型理论、圆柱阻力、螺旋桨推力、NACA 0012、RAE 2822），无需 CFD 即可验证代理模型管线 |
| **主动学习** | `ActiveLearner`——贝叶斯优化智能采样 |
| **多保真度** | `CoKriging`——融合低精度与高精度仿真数据 |
| **GPU/CUDA** | MLP 后端支持 `device='cuda'` |
| **多输出** | 一个代理模型同时预测 CL、CD、CM |
| **验证套件** | 交叉验证、学习曲线、残差分析、扩展指标（R²/RMSE/MAE/MaxRE/Explained Variance） |
| **物理约束** | 单调性、凸性、边界值、Sobolev 梯度约束——在训练时锁定物理规律 |
| **ONNX 导出** | 训练好的 MLP 代理模型可导出部署到任意 ONNX 运行时 |
| **Scikit-learn 风格** | `.fit()`、`.predict()`、`.validate()`——会用 sklearn 就会用 Prandtl |

## 快速上手

### 1. 解析函数验证（零 CFD）

```python
import prandtl as pr

# 薄翼型升力系数：CL = 2π(α + 2camber)
X, Y = pr.sample(
    pr.analytical.cl_flat_plate,
    bounds=[(-5, 15),     # α：攻角 -5° 到 15°
            (0.01, 0.1)],  # camber：弯度 1% 到 10%
    n=100,
    method="lhs",
    seed=42,
)

surrogate = pr.Surrogate(params=["alpha", "camber"], outputs=["CL"], method="gp")
surrogate.fit(X, Y)  # 学习解析函数

# 在新点上测试
X_test, Y_test = pr.sample(pr.analytical.cl_flat_plate, bounds=[(-5, 15), (0.01, 0.1)], n=30, seed=99)
Y_pred = surrogate.predict(X_test)
report = pr.metrics(Y_test, Y_pred)
print(report)  # 对光滑解析函数，R² > 0.999
```

### 2. 多输出

```python
def my_airfoil(alpha, mach):
    cl = 2 * np.pi * (np.radians(alpha) + 0.04)
    cd = 0.01 + 0.1 * cl**2  # 二次阻力极线
    return {"CL": cl, "CD": cd}

X, Y = pr.sample(my_airfoil, bounds=[(-5, 15), (0.15, 0.85)], n=200)

surrogate = pr.Surrogate(
    params=["alpha", "mach"], outputs=["CL", "CD"], method="mlp"
)
surrogate.fit(X, Y, n_iter=3000)

# 一次调用验证全部输出
Y_pred = surrogate.predict(X_test)
report = pr.metrics(Y_test, Y_pred)
# {"CL": {"r2": 0.9995, "rmse": ..., "mae": ...},
#  "CD": {"r2": 0.9987, "rmse": ..., "mae": ...}}
```

### 3. 导出 ONNX

```python
# MLP 代理模型可导出部署
surrogate.export("airfoil_model.onnx")
# 生成：airfoil_model__CL.onnx、airfoil_model__CD.onnx

# 用 onnxruntime 加载
import onnxruntime as ort
session = ort.InferenceSession("airfoil_model__CL.onnx")
cl = session.run(None, {"X": x_new.astype(np.float32)})[0]
```

### 4. 采样方法

```python
# 拉丁超立方采样（默认）—— 空间填充
X, Y = pr.sample(func, bounds=[(0, 1), (-2, 2)], n=100, method="lhs")

# 均匀随机采样
X, Y = pr.sample(func, bounds=[(0, 1), (-2, 2)], n=100, method="uniform")

# Sobol 序列 —— 低差异、可复现
X, Y = pr.sample(func, bounds=[(0, 1), (-2, 2)], n=128, method="sobol")

# 从已有数据训练
surrogate = pr.Surrogate(params=["alpha", "mach"], outputs=["CL"], method="gp")
surrogate.fit(X, Y)  # X: (n_points, n_params), Y: (n_points, n_outputs)
```

### 5. 物理信息训练

```python
from prandtl import Monotonicity, BoundaryValue, Convexity

constraints = [
    Monotonicity(param_idx=0, sign=1, weight=0.1),
    # CL 必须随 α（参数索引 0）单调增加。sign=+1 强制单调增。
    BoundaryValue({"alpha": 0.0}, {"CL": 0.0}, weight=10.0),
    # 在 α=0° 时 CL 必须为 0。高权重 = 严格约束。
    Convexity(param_idx=0, sign=-1, weight=0.05),
    # 凹关系（sign=-1）——例如阻力极线的曲率。
]

surrogate = pr.Surrogate(params=["alpha", "mach"], outputs=["CL", "CD"], method="mlp")
surrogate.fit(X, Y, physics=constraints, n_iter=500, lr=0.01)
```

### 6. 交叉验证

```python
# 5 折 CV：80% 训练，20% 测试，重复 5 次
scores = pr.cross_validate(surrogate, X, Y, cv=5, verbose=True)
print(f"MAE: {scores['CL']['mae_mean']:.4f} ± {scores['CL']['mae_std']:.4f}")
print(f"R²:  {scores['CL']['r2_mean']:.4f} ± {scores['CL']['r2_std']:.4f}")

# 所有输出自动评分
# {'CL': {'mae_mean': ..., 'mae_std': ..., 'rmse_mean': ..., 'r2_mean': ..., ...},
#  'CD': {'mae_mean': ..., ...}}
```

### 7. 学习曲线

```python
# 看模型性能随数据量如何变化
curve = pr.learning_curve(surrogate, X, Y, sizes=[10, 20, 50, 100, 150])

# 解读：若 val_mae 趋于平缓 → 数据量够用。
# 若 train_mae ≪ val_mae → 过拟合 → 试试更简单的模型或减少迭代。
print(f"最终训练 MAE: {curve['train_mae'][-1]:.4f}")
print(f"最终验证 MAE: {curve['val_mae'][-1]:.4f}")
```

### 8. 残差分析

```python
res = pr.residual_analysis(Y_test, Y_pred)

# Shapiro-Wilk 正态检验：p > 0.05 → 残差正态 ✓
for output in res:
    r = res[output]
    print(f"{output}:")
    print(f"  Shapiro-Wilk p={r['shapiro_p']:.3f} {'✓ 正态' if r['shapiro_p'] > 0.05 else '✗ 非正态'}")
    print(f"  偏度={r['skewness']:.3f}，峰度={r['kurtosis']:.3f}")
    print(f"  最大残差位置: index {r['max_residual_idx']}")

# 偏度高 → 系统偏差。峰度高 → 异常值。
# 残差非正态 → 模型漏了物理规律，或数据不够。
```

## 内置解析函数

全部返回精确数学值——用于框架验证，完全不需要 CFD 数据。

| 函数 | 公式 | 参数 |
|----------|---------|------------|
| `cl_flat_plate(alpha, camber)` | CL = 2π(α + 2camber) | α：攻角[°]，camber：弯度比 |
| `cd_cylinder(reynolds)` | 分段 Re 相关 CD | Re：雷诺数 |
| `thrust_propeller(rpm, diameter, pitch)` | T = CT·ρ·n²·D⁴ | rpm、直径[m]、桨距[m] |

## 架构

```
prandtl/
├── __init__.py          # 公共 API：Surrogate、sample()、cross_validate()、metrics()、...
├── _surrogate.py        # 核心 Surrogate 类（fit/predict/validate/export）
├── _gaussian.py         # GPyTorch ExactGP 封装
├── _neural.py           # PyTorch MLP 封装
├── _tree.py             # 随机森林 & 梯度提升（scikit-learn）
├── _active.py           # 主动学习 / 贝叶斯优化
├── _co_kriging.py       # 多保真 Co-Kriging
├── _sobolev.py          # Sobolev 梯度约束
├── _validate.py         # 交叉验证、学习曲线、残差分析、评估指标
├── _physics.py          # 物理约束（Monotonicity、Convexity、BoundaryValue）
├── _sampling.py         # LHS、均匀、Sobol 采样器
├── _io.py               # CFD 数据读写（OpenFOAM forces、SU2 history）
├── _analytical.py       # 解析真值函数（NACA0012、RAE2822、平板、圆柱、螺旋桨）
└── analytical.py        # 公共导出接口
```

## 命名由来

**Prandtl** 取自 **路德维希·普朗特**（Ludwig Prandtl，1875–1953）——现代流体力学之父、边界层理论和升力线理论的奠基人。以他命名，致敬他为 CFD 领域奠定的数学基础。

## 当前局限

- **GP 不支持 ONNX 导出**：高斯过程是非参数方法，推理依赖全部训练数据，无法导出 ONNX。如需可导出模型请使用 `method='mlp'`
- **树模型不支持 ONNX 导出**：RF/GB 模型（基于 scikit-learn）无法导出 ONNX。如需导出请使用 `method='mlp'`
- **GB 不确定度**：梯度提升的不确定度量化需要分位数回归。请直接使用 `GradientBoosting.fit_with_uncertainty()`
- **Co-Kriging 规模**：当前版本仅支持两个保真度级别

## 路线图

**已完成：**
- [x] GP + MLP + RF + GB 四种后端
- [x] 物理约束（Monotonicity、Convexity、BoundaryValue、Sobolev 梯度）
- [x] 验证套件（交叉验证、学习曲线、残差分析）
- [x] CFD 数据读写（OpenFOAM、SU2）
- [x] ONNX 导出（MLP）
- [x] GPU/CUDA 支持
- [x] 不确定度量化 API
- [x] 主动学习 / 贝叶斯优化
- [x] 解析基准函数（NACA 0012、RAE 2822、平板、圆柱、螺旋桨）
- [x] 多保真代理模型（Co-Kriging）

**中期（v0.6+）：**
- [ ] 多级 Co-Kriging（3+ 保真度级别）
- [ ] 自适应采样策略（期望改进 EI、上置信界 UCB）
- [ ] 模型可解释性工具（SHAP、部分依赖图）
- [ ] 大规模数据分布式训练

## 贡献

欢迎提交 Issue 和 PR。请确保 `pytest tests/` 全部通过。

## 许可证

MIT License · Copyright (c) 2026 goodisok
