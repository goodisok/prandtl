# Prandtl · 普朗特

CFD 代理模型工具包。几行代码训练气动预测的快速代理模型——API 风格和 scikit-learn 一样直觉。

[English](../README.md) | **中文**

```python
import prandtl as pr

# 参数空间采样 + 解析真值（零 CFD 依赖）
X, Y = pr.sample(pr.analytical.cl_flat_plate, bounds=[(-5, 15), (0.01, 0.1)], n=100)

# 训练高斯过程代理模型
surrogate = pr.Surrogate(params=["alpha", "camber"], outputs=["CL"], method="gp")
surrogate.fit(X, Y)

# 预测 + 验证
Y_pred = surrogate.predict(X_test)
report = surrogate.validate(X_test, Y_test)
print(report)  # {"CL": {"r2": 0.9998, "rmse": 0.0012, "max_error": 0.0033}}

# 导出部署
surrogate.export("model.onnx")  # 每个输出一个 .onnx 文件
```

## 这有什么用？

**问题**：你的无人机旋翼升力有多大？ → 跑一次 CFD 仿真，40 分钟。

**你其实想问 100 个不同转速-攻角组合** → 那是 40×100 = 66 小时。

Prandtl 的做法：从 100 个测点学出规律 → 其他 10,000 个组合**毫秒级预测**，误差 < 0.2%。

用大白话说：CFD 仿真相当于一个**昂贵的计算器**——每按一次花半小时。Prandtl 做的事是**克隆这个计算器**——克隆品秒出结果，跟原件几乎一样准。

## v0.2.0 新功能

**物理约束**：纯数据不够——在 5° 和 15° 攻角之间，模型可能预测出升力先涨后跌。加约束锁定物理规律：

```python
from prandtl import Monotonicity, BoundaryValue

constraints = [
    Monotonicity(param_idx=0, sign=1, weight=0.1),    # CL 随 α 单调增
    BoundaryValue({"alpha": 0.0}, {"CL": 0.0}, weight=10.0),  # α=0 时 CL=0
]

surrogate.fit(X, Y, physics=constraints, n_iter=500, lr=0.01)
```

**CFD 数据读写**：从仿真输出到训练就绪，一句话：

```python
from prandtl import read_foam_forces, read_su2_history

X, Y = read_foam_forces("postProcessing/forces/0/coefficient.dat")
# → 直接喂给 surrogate.fit(X, Y)
```

换个角度：这是「机器学习」最务实的用法——不是生成图片、不是聊天、不是推荐——就是**学一个函数，替代另一个太慢的函数**。

## 安装

```bash
pip install prandtl[gp]          # 高斯过程后端（GPyTorch）
pip install prandtl[mlp]         # 神经网络后端（PyTorch）
pip install prandtl[export]      # ONNX 导出支持
pip install prandtl[all]         # 全部安装
```

## 能做什么

Prandtl 让你用快速 ML 代理模型替代昂贵的 CFD 仿真——不用写任何 ML 模板代码。

| 特性 | 说明 |
|---------|------------|
| **零 CFD 依赖** | 内置解析真值函数（薄翼型理论、圆柱阻力、螺旋桨推力），无需 CFD 即可验证代理模型管线 |
| **双后端** | 高斯过程（`method='gp'`，基于 GPyTorch）和 MLP（`method='mlp'`，基于 PyTorch） |
| **多输出** | 一个代理模型同时预测 CL、CD、CM |
| **验证报告** | 一行代码出 R²、RMSE、最大误差 |
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
report = surrogate.validate(X_test, Y_test)
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
report = surrogate.validate(X_test, Y_test)
# {"CL": {"r2": 0.9995, "rmse": ..., "max_error": ...},
#  "CD": {"r2": 0.9987, "rmse": ..., "max_error": ...}}
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
├── __init__.py          # 公共 API：Surrogate、sample()
├── _surrogate.py        # 核心 Surrogate 类（fit/predict/validate/export）
├── _gaussian.py         # GPyTorch ExactGP 封装
├── _neural.py           # PyTorch MLP 封装
├── _sampling.py         # LHS、均匀、Sobol 采样器
├── _analytical.py       # 解析真值函数
└── analytical.py        # 公共导出接口
```

## 命名由来

**Prandtl** 取自 **路德维希·普朗特**（Ludwig Prandtl，1875–1953）——现代流体力学之父、边界层理论和升力线理论的奠基人。以他命名，致敬他为 CFD 领域奠定的数学基础。

## 当前局限

- **GP 不支持 ONNX 导出**：高斯过程是非参数方法，推理依赖全部训练数据，无法导出 ONNX。如需可导出模型请使用 `method='mlp'`
- **暂无多保真**：当前仅支持单保真度。多保真（Co-Kriging）在规划中
- **暂无物理约束**：纯数据驱动拟合。PINN 风格物理约束和 Sobolev 训练在规划中
- **仅 CPU**：PyTorch 支持 CUDA，但尚未针对 Prandtl 做 GPU 优化

## 路线图

- [ ] 物理信息正则化（用 PDE 残差作为损失项）
- [ ] 多保真代理模型（Co-Kriging）
- [ ] Sobolev 训练（梯度约束）
- [ ] 内置 2D 翼型参数化
- [ ] OpenFOAM 算例生成与解析
- [ ] Isaac Sim 力/力矩注入插件

## 贡献

欢迎提交 Issue 和 PR。请确保 `pytest tests/` 全部通过。

## 许可证

MIT License · Copyright (c) 2026 goodisok
