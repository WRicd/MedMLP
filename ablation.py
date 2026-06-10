"""
MedMLP 优化器消融实验 (Ablation Study)
========================================
在 tissuemnist 上对比三种配置：
  1. NumPy MLP + SGD Momentum
  2. NumPy MLP + Adam
  3. PyTorch MLP + Adam (Baseline)

生成三曲线对比图表到 notebooks/。
"""

import os
import sys
import time
import copy
import yaml
import numpy as np

from src.data_pipeline import MedMNISTDataset, get_batches
from src.layers import NumpyMLP
from src.optimizers import SGDMomentum, AdamOptimizer
from src.metrics import accuracy, macro_f1_score
from benchmarks.pytorch_mlp import train_pytorch_mlp


def load_config(config_path="configs/base_config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def train_numpy_track(config, optimizer_name, dataset, epochs):
    """
    训练一条 NumPy 赛道。

    Parameters
    ----------
    config : dict
    optimizer_name : str - 'sgd_momentum' or 'adam'
    dataset : MedMNISTDataset
    epochs : int

    Returns
    -------
    results : dict
    """
    hidden_layers = config['model']['hidden_layers']
    batch_size = config['training']['batch_size']
    lr = config['training']['learning_rate']
    l2_lambda = config['training']['l2_lambda']

    # 构建网络（每次重新初始化）
    np.random.seed(42)
    model = NumpyMLP(dataset.input_dim, hidden_layers, dataset.num_classes)

    # 创建优化器
    if optimizer_name == 'sgd_momentum':
        optimizer = SGDMomentum(lr, momentum=config['sgd']['momentum'])
        label = "NumPy-SGD"
    elif optimizer_name == 'adam':
        optimizer = AdamOptimizer(
            lr,
            beta1=config['adam']['beta1'],
            beta2=config['adam']['beta2'],
            epsilon=config['adam']['epsilon'],
        )
        label = "NumPy-Adam"
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    print(f"\n{'─' * 60}")
    print(f"  [{label}] 开始训练")
    print(f"  网络: {dataset.input_dim} -> {' -> '.join(map(str, hidden_layers))} -> {dataset.num_classes}")
    print(f"  优化器: {optimizer_name} | LR: {lr} | Epochs: {epochs}")
    print(f"{'─' * 60}")

    train_losses = []
    val_accuracies = []
    epoch_times = []

    for epoch in range(epochs):
        epoch_start = time.time()

        # 训练
        epoch_loss = 0.0
        num_batches = 0
        for X_batch, Y_batch in get_batches(
            dataset.train_X, dataset.train_Y, batch_size, shuffle=True
        ):
            Z = model.forward(X_batch)
            loss = model.compute_loss(Z, Y_batch, l2_lambda=l2_lambda)
            model.backward(Y_batch, l2_lambda=l2_lambda)
            for layer in model.dense_layers:
                optimizer.update(layer, layer.dW, layer.db)
            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)

        # 验证
        val_preds = model.predict(dataset.val_X)
        val_true = np.argmax(dataset.val_Y, axis=0)
        val_acc = accuracy(val_preds, val_true)
        val_accuracies.append(val_acc)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        print(f"  Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | Time: {epoch_time:.2f}s")

    # 测试集评估
    test_preds = model.predict(dataset.test_X)
    test_true = np.argmax(dataset.test_Y, axis=0)
    test_acc = accuracy(test_preds, test_true)
    test_f1 = macro_f1_score(test_preds, test_true, dataset.num_classes)

    print(f"\n  [{label}] 测试准确率: {test_acc:.4f} | Macro F1: {test_f1:.4f}")

    return {
        'label': label,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'epoch_times': epoch_times,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
    }


def plot_ablation(results_list, pytorch_results, config, epochs):
    """绘制三曲线消融对比图。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figures_dir = config['output']['figures_dir']
    os.makedirs(figures_dir, exist_ok=True)
    dataset_name = config['data']['dataset']
    epoch_range = range(1, epochs + 1)

    # 统一调色板
    colors = {
        'NumPy-SGD': '#2196F3',    # 蓝
        'NumPy-Adam': '#4CAF50',   # 绿
        'PyTorch-Adam': '#FF5722', # 红
    }
    markers = {
        'NumPy-SGD': 'o',
        'NumPy-Adam': 's',
        'PyTorch-Adam': '^',
    }

    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': '#f8f9fa',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'font.size': 12,
    })

    # ===== 图1: Loss 三曲线对比 =====
    fig, ax = plt.subplots(figsize=(12, 7))

    for r in results_list:
        ax.plot(epoch_range, r['train_losses'],
                f"{markers[r['label']]}-", label=r['label'],
                color=colors[r['label']], linewidth=2, markersize=5)

    # PyTorch
    ax.plot(epoch_range, pytorch_results['train_losses'],
            f"{markers['PyTorch-Adam']}-", label='PyTorch-Adam',
            color=colors['PyTorch-Adam'], linewidth=2, markersize=5)

    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Training Loss', fontsize=14)
    ax.set_title(f'Optimizer Ablation Study - Loss ({dataset_name})',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='upper right')
    ax.set_xlim(1, epochs)
    fig.tight_layout()
    path = os.path.join(figures_dir, 'loss_comparison.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n[图表] Loss 消融对比: {path}")

    # ===== 图2: Accuracy 三曲线对比 =====
    fig, ax = plt.subplots(figsize=(12, 7))

    for r in results_list:
        ax.plot(epoch_range, r['val_accuracies'],
                f"{markers[r['label']]}-", label=r['label'],
                color=colors[r['label']], linewidth=2, markersize=5)

    ax.plot(epoch_range, pytorch_results['val_accuracies'],
            f"{markers['PyTorch-Adam']}-", label='PyTorch-Adam',
            color=colors['PyTorch-Adam'], linewidth=2, markersize=5)

    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Validation Accuracy', fontsize=14)
    ax.set_title(f'Optimizer Ablation Study - Accuracy ({dataset_name})',
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right')
    ax.set_xlim(1, epochs)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    path = os.path.join(figures_dir, 'accuracy_evaluation.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[图表] Accuracy 消融对比: {path}")


def main():
    print("=" * 70)
    print("  MedMLP 优化器消融实验 (Ablation Study)")
    print("=" * 70)

    config = load_config()
    dataset_name = config['data']['dataset']
    ablation_epochs = 10

    print(f"\n数据集: {dataset_name}")
    print(f"消融 Epoch 数: {ablation_epochs}")

    # 加载数据（只加载一次）
    dataset = MedMNISTDataset(dataset_name, config['data']['data_dir'])

    # ===== Track 1: NumPy + SGD Momentum =====
    sgd_results = train_numpy_track(config, 'sgd_momentum', dataset, ablation_epochs)

    # ===== Track 2: NumPy + Adam =====
    adam_results = train_numpy_track(config, 'adam', dataset, ablation_epochs)

    # ===== Track 3: PyTorch + Adam =====
    print(f"\n{'─' * 60}")
    print(f"  [PyTorch-Adam] 开始训练")
    print(f"{'─' * 60}")

    # 临时修改 config 的 epochs 为消融值
    config_pt = copy.deepcopy(config)
    config_pt['training']['epochs'] = ablation_epochs
    config_pt['output']['print_every'] = 1
    pytorch_results = train_pytorch_mlp(config_pt)

    # ===== 对比总结表 =====
    all_tracks = [sgd_results, adam_results,
                  {'label': 'PyTorch-Adam',
                   'test_accuracy': pytorch_results['test_accuracy'],
                   'test_f1': pytorch_results['test_f1'],
                   'epoch_times': pytorch_results['epoch_times']}]

    print(f"\n{'=' * 70}")
    print(f"  消融实验总结 ({dataset_name}, {ablation_epochs} Epochs)")
    print(f"{'=' * 70}")
    print(f"  {'配置':<20} {'Test Acc':>10} {'Macro F1':>10} {'总耗时':>10}")
    print(f"  {'-' * 55}")
    for t in all_tracks:
        total_time = sum(t['epoch_times'])
        print(f"  {t['label']:<20} {t['test_accuracy']:>10.4f} {t['test_f1']:>10.4f} {total_time:>8.2f}s")

    # ===== 绘制三曲线图 =====
    plot_ablation([sgd_results, adam_results], pytorch_results, config, ablation_epochs)

    print(f"\n{'=' * 70}")
    print(f"  消融实验完成！")
    print(f"{'=' * 70}")

    # 返回结果供 README 使用
    return {
        'sgd': sgd_results,
        'adam': adam_results,
        'pytorch': pytorch_results,
    }


if __name__ == '__main__':
    main()
