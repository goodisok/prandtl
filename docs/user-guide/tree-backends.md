# 树模型后端

Random Forest 和 Gradient Boosting 后端基于 scikit-learn，**无需安装 PyTorch 或 GPyTorch**。轻量、快速，适合基线建模和快速迭代。

## 安装

```bash
pip install prandtl-cfd[tree]
```

或直接安装 scikit-learn：

```bash
pip install scikit-learn
```

## Random Forest

集成多棵决策树，通过平均预测值减少过拟合。训练快、鲁棒性强。

```python
import prandtl as pr

# 采样
X, Y = pr.sample(
    pr.analytical.cl_flat_plate,
    bounds=[(-5, 15), (0.01, 0.1)],
    n=200, method="lhs", seed=42
)

# 训练 Random Forest
surrogate = pr.Surrogate(
    params=["alpha", "camber"],
    outputs=["CL"],
    method="rf"
)
surrogate.fit(X, Y)

# 预测
Y_pred = surrogate.predict(X_test)

# RF 特有：不确定性估计
Y_mu, Y_std = surrogate.predict_with_uncertainty(X_test)
print(f"预测均值: {Y_mu[:5]}")
print(f"预测标准差: {Y_std[:5]}")  # 树集成方差 → 不确定性
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_estimators` | 100 | 树的数量 |
| `max_depth` | None | 树的最大深度 |

## Gradient Boosting

逐步构建弱学习器，每个新树纠正前一棵的残差，精度通常更高。

```python
surrogate = pr.Surrogate(
    params=["alpha", "mach"],
    outputs=["CL", "CD"],
    method="gb"
)
surrogate.fit(X, Y)
Y_pred = surrogate.predict(X_test)
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_estimators` | 100 | 提升阶段数 |
| `learning_rate` | 0.1 | 每棵树的贡献衰减 |
| `max_depth` | 3 | 树的最大深度 |

## predict_with_uncertainty()

仅 Random Forest 支持。返回预测均值和标准差（树集成方差）：

```python
Y_mu, Y_std = surrogate.predict_with_uncertainty(X_test)

# Y_mu:  模型预测均值（与 predict() 相同）
# Y_std: 树与树之间的标准差 → 不确定性指标
#        值越大，模型对该点越不确定
```

!!! note "GB 不确定性"
    Gradient Boosting 的标准实现不直接提供不确定性。如需 GB 的预测区间，可改用分位数回归或结合其他方法。

## 何时选择树模型

- 不想安装 PyTorch / GPyTorch 依赖
- 中等数据量（几百到几千样本）
- 需要快速原型开发
- 需要 RF 的不确定性估计
- GPU 不可用
