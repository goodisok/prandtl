# GPU 训练

在 GPU 上加速 MLP 后端训练，适合大规模数据集。

## 用法

```python
import prandtl as pr

surrogate = pr.Surrogate(
    params=["alpha", "mach"],
    outputs=["CL", "CD"],
    method="mlp",
    device="cuda"  # 启用 GPU 训练
)
surrogate.fit(X, Y, n_iter=5000, lr=0.001)
```

只需设置 `device="cuda"`，模型和数据会自动转移到 GPU。默认值为 `"cpu"`。

## 何时使用

| 场景 | 建议 |
|------|------|
| 样本数 < 1000 | CPU 即可，通信开销可能抵消 GPU 加速 |
| 样本数 1000–10000 | GPU 明显加速（2–10×） |
| 样本数 > 10000 | **强烈推荐** GPU |
| 仅用 GP / 树模型 | GPU 无效（GPyTorch 可使用 `device="cuda"`，但收益有限） |

## 注意事项

- **仅 MLP 受益最大**：树模型（RF/GB）基于 scikit-learn，不支持 GPU。
- **GPyTorch 也支持 CUDA**：`method="gp"` 配合 `device="cuda"` 可用，但对中小数据集加速有限。
- **需要 CUDA 环境**：确保安装 PyTorch CUDA 版本。可通过 `torch.cuda.is_available()` 验证。
- **内存开销**：GPU 显存有限，超大模型或超大批次可能需要 `batch_size` 参数调节。

```python
# 检查 CUDA 是否可用
import torch
print(torch.cuda.is_available())  # True → 可以使用 device="cuda"
```
