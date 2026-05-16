# 主动学习

主动学习自动选择"下一个最有价值的采样点"，用最少的 CFD 仿真达到最佳代理模型精度。

## propose_next() — 单步建议

```python
from prandtl import propose_next

# surrogate 必须是已训练的 GP 模型（MLP 不支持不确定性估计）
X_next = propose_next(
    surrogate,
    bounds=[(-5, 15), (0.01, 0.1)],
    strategy="不确定性策略"
)
```

每次调用返回**一个**建议采样点 `(n_params,)`。需要自己评估真实函数、合并数据、重新训练。

## 采样策略

| 策略 | 说明 | 需要 y_best |
|------|------|-------------|
| `"uncertainty"` | 最大预测方差 | 否 |
| `"ei"` | Expected Improvement（最小化问题） | 是 |
| `"ucb"` | Upper Confidence Bound | 否 |
| `"pi"` | Probability of Improvement | 是 |

!!! warning "仅限 GP"
    以上策略依赖预测方差，只能用于 `method="gp"` 的代理模型。

## active_learn() — 全自动循环

```python
from prandtl import active_learn, Surrogate

def my_cfd(alpha, camber):
    """真实函数 — 实际应用中替换为 CFD 求解器调用"""
    import numpy as np
    cl = np.sin(np.radians(alpha)) * (1 + camber * 10)
    return {"CL": cl}

surrogate = Surrogate(params=["alpha", "camber"], outputs=["CL"], method="gp")

X, Y, history = active_learn(
    my_cfd,
    bounds=[(-5, 15), (0.01, 0.1)],
    surrogate=surrogate,
    n_initial=10,      # 初始随机样本数
    n_iter=10,          # 主动学习轮数
    strategy="ei",
    seed=42,
    verbose=True
)
```

参数说明：

| 参数 | 说明 |
|------|------|
| `func` | 真实函数，接受关键字参数（如 `alpha=5, camber=0.04`），返回 `{"output": value}` 字典 |
| `bounds` | 参数范围列表 |
| `surrogate` | **未训练**的 GP `Surrogate` 实例（作为配置模板） |
| `n_initial` | 初始随机采样数（默认 10） |
| `n_iter` | 主动学习迭代轮数（默认 10） |
| `strategy` | 采集策略：`"uncertainty"` / `"ei"` / `"ucb"` / `"pi"` |

返回值：
- `X` — 全部采样点 `(n_total, n_params)`
- `Y` — 全部函数值 `(n_total, n_outputs)`
- `history` — 每轮的 best Y 值追踪

## 手动循环

```python
import prandtl as pr
import numpy as np

X, Y = pr.sample(pr.analytical.cl_flat_plate,
                 bounds=[(-5, 15), (0.01, 0.1)], n=10, method="lhs")
surr = pr.Surrogate(params=["alpha", "camber"], outputs=["CL"], method="gp").fit(X, Y)

for i in range(5):
    x_next = pr.propose_next(surr, bounds=[(-5, 15), (0.01, 0.1)], strategy="uncertainty")
    _, y_next = pr.sample(pr.analytical.cl_flat_plate,
                          bounds=[(-5, 15), (0.01, 0.1)], n=1, method="lhs")
    X = np.vstack([X, x_next.reshape(1, -1)])
    Y = np.vstack([Y, y_next])
    surr.fit(X, Y)
```

!!! note "与实际仿真集成"
    `y_next` 的位置替换为真实 CFD 求解器调用。Prandtl 不执行仿真，只告诉你**应该在哪里跑仿真**。
