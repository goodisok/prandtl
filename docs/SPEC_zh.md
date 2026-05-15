# 设计文档：Prandtl — CFD 代理模型工具包

[English](SPEC.md) | **中文**

## 目标

**是什么**：一个 Python 工具包，让仿真工程师用最少的代码训练、验证和导出 CFD 代理模型。三行代码：采样 → 拟合 → 导出。

**给谁用**：需要快速气动预测而不想每次都跑完整 CFD 的仿真工程师。

**为什么做**：现有方案（scikit-learn GP、SMT）是通用 ML 工具，没有一个提供「参数采样 → 代理训练 → 验证报告 → 仿真器可读导出」的完整领域工作流。每家航空航天/机器人公司都在内部造轮子，开源领域没有标准。

**成功标准**：工程师用训练好的代理模型替代 CFD 仿真循环，在亚毫秒延迟下预测 CL/CD/CM，留出数据 R² > 0.95。

## 技术栈

| 组件 | 选择 | 版本 |
|-----------|--------|---------|
| 语言 | Python | ≥ 3.10 |
| 高斯过程 | GPyTorch | latest |
| 神经网络 | PyTorch | latest |
| 导出 | ONNX + onnxruntime | latest |
| 采样 | scipy（LHS） | latest |
| 数学 | numpy | latest |
| 构建 | setuptools / pyproject.toml | PEP 621 |
| 测试 | pytest | latest |
| 代码检查 | ruff | latest |

## 常用命令

```
安装：     pip install -e .
测试：     pytest tests/ -v
代码检查： ruff check src/
格式化：   ruff format src/
类型检查： mypy src/
```

## 项目结构

```
prandtl/
├── pyproject.toml          # PEP 621 构建配置
├── README.md               # 英文文档
├── README_zh.md            # 中文文档
├── docs/
│   ├── SPEC.md             # 本文件（英文）
│   └── SPEC_zh.md          # 本文件（中文）
├── src/
│   └── prandtl/
│       ├── __init__.py     # 公共 API
│       ├── _sampling.py    # LHS、均匀、Sobol 采样器
│       ├── _analytical.py  # 解析真值函数（用于验证）
│       ├── _surrogate.py   # 核心 Surrogate 类（统一接口）
│       ├── _gaussian.py    # GP 后端（GPyTorch）
│       └── _neural.py      # MLP 后端（PyTorch）
└── tests/
    └── test_e2e.py         # 端到端测试（23 个）
```

下划线前缀为私有模块。仅 `__init__.py` 暴露公共 API。

## 代码风格

```python
"""单行模块文档字符串。"""

from typing import Optional

import numpy as np
import torch


class Surrogate:
    """CFD 代理模型，接口风格类似 scikit-learn。

    Parameters
    ----------
    params : list of str
        输入参数名称，例如 ['alpha', 'mach', 'camber']。
    outputs : list of str
        输出量名称，例如 ['CL', 'CD']。
    method : str
        后端选择：'gp'（高斯过程）或 'mlp'（神经网络）。
    """

    def __init__(
        self,
        params: list[str],
        outputs: list[str],
        method: str = "gp",
    ) -> None:
        ...

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        *,
        n_iter: int = 100,
        verbose: bool = True,
    ) -> "Surrogate":
        """在 (X, Y) 数据上训练代理模型。返回 self 以支持链式调用。"""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """对给定输入返回预测输出。"""
        ...

    def validate(self, X_test: np.ndarray, Y_test: np.ndarray) -> dict:
        """返回各输出的 R²、RMSE、max_error 字典。"""
        ...
```

关键约定：
- Google 风格文档字符串（面向科学计算的 numpy docstring 格式）
- 所有公共方法带类型标注
- 私有模块以 `_` 为前缀
- 类名：PascalCase。函数/变量：snake_case
- 行宽限制 100 字符
- 适当使用显式 `*` 标记仅关键字参数

## 测试策略

- **框架**：pytest，启用 `--strict-markers`
- **位置**：`tests/` 对应 `src/prandtl/`
- **覆盖率目标**：核心模块 > 90%
- **测试层级**：
  - 单元测试：每个模块独立测试，使用小规模合成数据
  - 集成测试：`Surrogate.fit() → .predict() → .validate()` 端到端，使用解析真值
- **无需 GPU** —— 全部测试可在 CPU 上用小规模合成数据运行
- **CI**（未来）：GitHub Actions，push 触发，仅 CPU

## API 设计（MVP）

### 三行接口

```python
import prandtl as pr

# 1. 采样
X, Y = pr.sample(pr.analytical.cl_flat_plate, bounds=[(-5, 15), (0.01, 0.1)], n=100)

# 2. 拟合
surrogate = pr.Surrogate(params=["alpha", "camber"], outputs=["CL"]).fit(X, Y)

# 3. 验证
report = surrogate.validate(*pr.sample(pr.analytical.cl_flat_plate, bounds=[...], n=20))
print(f"R² = {report['CL']['r2']:.4f}")  # → R² = 0.9998
```

### 模块分解

#### `prandtl.sample(func, bounds, n, method='lhs')`
采样参数空间并计算真值函数。
- `func`：接受 `**params` 并返回输出字典的可调用对象
- `bounds`：每个参数的 (下限, 上限) 元组列表
- `n`：设计点数量
- `method`：'lhs' | 'uniform' | 'sobol'
- 返回：`(X: np.ndarray, Y: np.ndarray)`

#### `prandtl.Surrogate(params, outputs, method='gp')`
主类。GP 和 MLP 后端的统一接口。

#### `analytical` 模块
用于框架验证的内置真值函数：
- `cl_flat_plate(alpha, camber)` → CL = 2π(α + 2c)  [薄翼型理论]
- `cd_cylinder(reynolds)` → 经验阻力曲线
- `thrust_propeller(rpm, diameter, pitch)` → T = CT·ρ·n²·D⁴

## MVP 成功标准

- [x] `pip install -e .` 成功
- [x] `pr.sample()` 在 LHS 和均匀方法下返回正确形状
- [x] GP 代理模型在 100 训练/20 测试点上拟合 `cl_flat_plate`，R² > 0.99
- [x] MLP 代理模型在 100 训练/20 测试点上拟合 `cl_flat_plate`，R² > 0.99
- [x] `surrogate.validate()` 返回各输出的 r2、rmse、max_error 字典
- [x] `surrogate.export('model.onnx')` 生成有效 ONNX 文件
- [x] 全部测试通过：`pytest tests/ -v`
- [x] Ruff 无告警：`ruff check src/`

## 边界规则

**必须做：**
- 声称功能完成前运行测试
- 所有公共函数和类写文档字符串
- 所有公共 API 加类型标注
- 保持导入最小化——无不使用的依赖

**先问再改：**
- 添加 numpy、scipy、torch、gpytorch、onnx 以外的新依赖
- 修改公共 API（`__init__.py` 中的任何内容）
- 添加 GPU 相关代码路径

**禁止做：**
- 硬编码文件路径
- 假定运行时有网络连接
- 在模块级别导入重依赖（`__init__.py` 中惰性导入）

## 待解决问题

无——已全部在以上假设中解决。
