# 示例：树模型代理

使用 Random Forest 对平板翼型升力系数建模，并利用不确定性估计评估预测可靠性。

## 完整代码

```python
import prandtl as pr
import numpy as np

# 1. 采样：平板翼型升力 CL = 2π(α + 2camber)
X, Y = pr.sample(
    pr.analytical.cl_flat_plate,
    bounds=[(-5, 15), (0.01, 0.1)],
    n=200, method="lhs", seed=42
)

# 2. 训练 Random Forest 代理模型
surrogate = pr.Surrogate(
    params=["alpha", "camber"],
    outputs=["CL"],
    method="rf",
    n_estimators=200,
    max_depth=10
)
surrogate.fit(X, Y)

# 3. 生成测试点并预测
X_test, Y_test = pr.sample(
    pr.analytical.cl_flat_plate,
    bounds=[(-5, 15), (0.01, 0.1)],
    n=50, method="lhs", seed=99
)
Y_pred = surrogate.predict(X_test)

# 4. 不确定性估计
Y_mu, Y_std = surrogate.predict_with_uncertainty(X_test)

# 5. 评估
report = pr.metrics(Y_test, Y_pred)
print(f"R²: {report['CL']['r2']:.6f}")
print(f"RMSE: {report['CL']['rmse']:.6f}")
print(f"MAE: {report['CL']['mae']:.6f}")

# 6. 找出最不确定的点
max_std_idx = np.argmax(Y_std)
print(f"\n最大不确定度: {Y_std[max_std_idx][0]:.6f}")
print(f"对应参数: alpha={X_test[max_std_idx, 0]:.1f}°, camber={X_test[max_std_idx, 1]:.3f}")
print(f"预测 CL: {Y_mu[max_std_idx][0]:.4f} ± {Y_std[max_std_idx][0]:.4f}")

# 7. 交叉验证
scores = pr.cross_validate(surrogate, X, Y, cv=5)
print(f"\n5折CV MAE: {scores['CL']['mae_mean']:.6f} ± {scores['CL']['mae_std']:.6f}")
```

## 输出解读

- **R² ≈ 0.999**：RF 在平滑函数上表现极佳
- **Y_std**：标准差在训练数据稀疏区域较大 → 指导下一轮采样
- **无需 PyTorch**：整个流程只需 numpy + scipy + scikit-learn

## RF vs GP

| 维度 | Random Forest | Gaussian Process |
|------|---------------|------------------|
| 安装依赖 | scikit-learn | GPyTorch |
| 训练速度 | 极快 | O(n³) |
| 不确定性 | 树集成方差（启发式） | 解析后验方差 |
| 大数据 | 友好 | 受限 |
| ONNX 导出 | 不支持 | 不支持 |
