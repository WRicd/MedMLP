# MedMLP — 解耦型双轨机器学习框架

## 项目目录

```text
MedMLP/
├── configs/
│   └── base_config.yaml       # 超参数配置（YAML 驱动）
├── src/                       # 自研 NumPy 引擎核心
│   ├── __init__.py
│   ├── data_pipeline.py       # 数据流水线（自适应扁平化 + Mini-batch）
│   ├── layers.py              # 矩阵算子（DenseLayer / ReLU / SoftmaxCE）
│   ├── optimizers.py          # 优化器（SGD Momentum / Adam）
│   └── metrics.py             # 评估指标（Accuracy / Macro F1 / 混淆矩阵）
├── benchmarks/
│   └── pytorch_mlp.py         # PyTorch 对标基准
├── data/                      # MedMNIST .npz 数据文件
│   ├── bloodmnist.npz
│   ├── pathmnist.npz
│   ├── octmnist.npz
│   └── tissuemnist.npz
├── notebooks/                 # 自动生成的对比图表
│   ├── loss_comparison.png
│   └── accuracy_evaluation.png
├── legacy/                    # 历史版本（隔离归档）
├── train.py                   # 主训练入口（双轨并行）
├── ablation.py                # 优化器消融实验
├── requirements.txt
└── README.md
```

---

## 支持的 MedMNIST 数据集

| 数据集 | 任务 | 通道 | 输入维度 | 类别数 | 样本总量 |
|--------|------|------|----------|--------|----------|
| `bloodmnist` | 外周血细胞分类 | 3 (RGB) | 2352 | 8 | 17,092 |
| `pathmnist` | 结直肠癌病理分类 | 3 (RGB) | 2352 | 9 | 107,081 |
| `octmnist` | 视网膜 OCT 分类 | 1 (灰度) | 784 | 4 | 109,309 |
| `tissuemnist` | 组织细胞多分类 | 1 (灰度) | 784 | 8 | 236,528 |

---

## 性能对比

### BloodMNIST 冒烟测试（30 Epochs）

| 指标 | NumPy MLP | PyTorch MLP | 差距 |
|------|-----------|-------------|------|
| **测试准确率** | **0.8372** | 0.8308 | +0.64% |
| **Macro F1** | **0.8106** | 0.8071 | +0.34% |
| 总训练耗时 | 102.55s | 5.87s (CUDA) | — |

> 准确率差距仅 **0.64%**，远在 ±2% 达标阈值以内。

### TissueMNIST 压测 — 优化器消融实验（10 Epochs, 165,466 训练样本）

| 配置 | 测试准确率 | Macro F1 | 总耗时 |
|------|-----------|----------|--------|
| **NumPy-Adam** | **0.5681** | **0.3906** | 161.46s |
| PyTorch-Adam | 0.5628 | 0.3895 | 43.50s (CUDA) |
| NumPy-SGD | 0.4647 | 0.2006 | 101.69s |

> **NumPy-Adam vs PyTorch-Adam 差距仅 +0.53%**，再次验证自研引擎的正确性。
>
> **Adam 优化器显著碾压 SGD**：准确率提升 +10.34%，F1 提升 +19.00%，充分体现了自适应学习率在高维复杂任务中的优势。
>
> **压测通过**：纯 NumPy 引擎在 16.5 万级数据上稳定运行，无内存崩溃、无 NaN，每 Epoch 约 15 秒。

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 一键训练（双轨对标）

```bash
python train.py
```

默认使用 `configs/base_config.yaml` 中的配置。训练完成后自动在 `notebooks/` 下生成对比图表。

### 3. 切换数据集

编辑 `configs/base_config.yaml`：

```yaml
data:
  dataset: "bloodmnist"   # 可选: bloodmnist, pathmnist, octmnist, tissuemnist
```

### 4. 运行消融实验

```bash
python ablation.py
```

在当前配置的数据集上对比 NumPy-SGD / NumPy-Adam / PyTorch-Adam 三种配置。

---

## 超参数配置

所有训练参数通过 `configs/base_config.yaml` 集中管理，零硬编码：

```yaml
model:
  hidden_layers: [256, 128]    # 隐藏层拓扑
training:
  epochs: 30
  batch_size: 128
  learning_rate: 0.001
  optimizer: "adam"            # adam / sgd_momentum
  l2_lambda: 0.0001           # L2 正则化系数
```

---

## 数学公式

### 前向传播
$$Z^{[l]} = W^{[l]} X^{[l-1]} + b^{[l]}, \quad A^{[l]} = \text{ReLU}(Z^{[l]})$$

### 反向传播（含 L2 正则化）
$$\frac{\partial \mathcal{L}}{\partial W} = \frac{1}{N} dZ \cdot X^T + \frac{\lambda}{N} W$$

### Adam 优化器
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$W = W - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

---

## 架构设计

```
输入层 (D)  →  Dense(D, 256) → ReLU → Dense(256, 128) → ReLU → Dense(128, C) → Softmax+CE
    ↑                                                                              ↓
  (D,N)                                                                       Loss + dZ
列向量模式                                                                    反向传播 ←
```

**核心约定**：全局使用列向量模式 `X: (D, N)`，保证矩阵求导公式紧凑统一。

---

## License

本项目仅用于学术研究与学习目的。
