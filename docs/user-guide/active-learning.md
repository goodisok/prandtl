# 主动学习

主动学习自动选择"下一个最有价值的采样点"，用最少的 CFD 仿真达到最佳代理模型精度。

## ActiveLearner

```python
from prandtl import ActiveLearner

learner = ActiveLearner(
    surrogate,       # 已训练的代理模型
    X_pool,          # 候选池 (n_pool, n_params)
    strategy="max_std"  # 采样策略
)
X_next = learner.query(n=10)  # 选出 10 个最不确定的点
```

## 采样策略

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `"max_std"` | 选择模型最不确定的点（最大标准差） | GP、RF 等有不确定性估计的后端 |
| `"random"` | 从候选池中随机采样 | 基线对比、探索初期 |

`"max_std"` 要求后端支持 `predict_with_uncertainty()` — 目前 GP 和 RF 支持。

## 完整工作流

```python
import prandtl as pr
from prandtl import ActiveLearner
import numpy as np

# 1. 初始采样（少量）
X_init, Y_init = pr.sample(
    pr.analytical.cl_flat_plate,
    bounds=[(-5, 15), (0.01, 0.1)],
    n=20, method="lhs", seed=42
)

# 2. 创建候选池（大量未标注点）
X_pool, _ = pr.sample(
    pr.analytical.cl_flat_plate,
    bounds=[(-5, 15), (0.01, 0.1)],
    n=500, method="lhs", seed=99
)

# 3. 初始训练
surrogate = pr.Surrogate(
    params=["alpha", "camber"],
    outputs=["CL"],
    method="gp"
)
surrogate.fit(X_init, Y_init)

# 4. 主动学习循环
learner = ActiveLearner(surrogate, X_pool, strategy="max_std")

for iteration in range(5):
    # 查询 5 个最不确定的点
    X_query = learner.query(n=5)

    # 标注（用真实函数或 CFD 仿真）
    _, Y_query = pr.sample(
        pr.analytical.cl_flat_plate,
        bounds=[(-5, 15), (0.01, 0.1)],
        n=len(X_query), method="lhs", seed=100+iteration
    )

    # 合并数据并重新训练
    X_all = np.vstack([X_init, X_query])
    Y_all = np.vstack([Y_init, Y_query])
    surrogate.fit(X_all, Y_all)

    # 从候选池中移除已查询的点
    learner.remove_queried(X_query)

    print(f"Iteration {iteration+1}: queried {len(X_query)} points")

# 5. 最终评估
Y_pred = surrogate.predict(X_pool)
```

## query() 与 remove_queried()

- `query(n)` — 从候选池中选出 `n` 个最不确定的点。
- `remove_queried(X)` — 从候选池中移除已查询的点，避免重复采样。

!!! note "候选池与 CFD 集成"
    实际应用中，`X_query` 是真正需要运行 CFD 求解器的参数点。Prandtl 不执行仿真，只告诉你*应该在哪里仿真*。
