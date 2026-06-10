"""
MedMLP 双轨训练入口
=====================
读取 configs/base_config.yaml 驱动整个训练流程：
1. 先跑自研 NumPy MLP，记录 Train Loss / Val Accuracy / 耗时
2. 再跑 PyTorch Baseline，记录相同指标
3. 使用 matplotlib 绘制对比图表到 notebooks/
"""

import os
import sys
import time
import yaml
import numpy as np

from src.data_pipeline import MedMNISTDataset, get_batches
from src.layers import NumpyMLP
from src.optimizers import SGDMomentum, AdamOptimizer
from src.metrics import accuracy, macro_f1_score, confusion_matrix
from benchmarks.pytorch_mlp import train_pytorch_mlp


def load_config(config_path="configs/base_config.yaml"):
    """加载 YAML 配置文件。"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def create_optimizer(config, model):
    """根据配置创建优化器。"""
    opt_name = config['training']['optimizer']
    lr = config['training']['learning_rate']

    if opt_name == 'sgd_momentum':
        momentum = config['sgd']['momentum']
        return SGDMomentum(lr, momentum=momentum)
    elif opt_name == 'adam':
        return AdamOptimizer(
            lr,
            beta1=config['adam']['beta1'],
            beta2=config['adam']['beta2'],
            epsilon=config['adam']['epsilon'],
        )
    else:
        raise ValueError(f"未知优化器: {opt_name}")


def train_numpy_mlp(config):
    """
    训练自研 NumPy MLP。

    Returns
    -------
    results : dict
        train_losses, val_accuracies, epoch_times, test_accuracy, test_f1
    """
    dataset_name = config['data']['dataset']
    data_dir = config['data']['data_dir']
    hidden_layers = config['model']['hidden_layers']
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    l2_lambda = config['training']['l2_lambda']
    print_every = config['output']['print_every']

    # 加载数据
    dataset = MedMNISTDataset(dataset_name, data_dir)

    # 构建网络
    model = NumpyMLP(dataset.input_dim, hidden_layers, dataset.num_classes)
    optimizer = create_optimizer(config, model)

    print(f"\n[NumPy MLP] 开始训练")
    print(f"  网络结构: {dataset.input_dim} -> {' -> '.join(map(str, hidden_layers))} -> {dataset.num_classes}")
    print(f"  优化器: {config['training']['optimizer']} | LR: {config['training']['learning_rate']}")
    print(f"  Epochs: {epochs} | Batch Size: {batch_size} | L2: {l2_lambda}")

    train_losses = []
    val_accuracies = []
    epoch_times = []

    for epoch in range(epochs):
        epoch_start = time.time()

        # --- 训练阶段 ---
        epoch_loss = 0.0
        num_batches = 0

        for X_batch, Y_batch in get_batches(
            dataset.train_X, dataset.train_Y, batch_size, shuffle=True
        ):
            # 前向传播
            Z = model.forward(X_batch)
            loss = model.compute_loss(Z, Y_batch, l2_lambda=l2_lambda)

            # 反向传播
            model.backward(Y_batch, l2_lambda=l2_lambda)

            # 参数更新
            for layer in model.dense_layers:
                optimizer.update(layer, layer.dW, layer.db)

            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        train_losses.append(avg_loss)

        # --- 验证阶段 ---
        val_preds = model.predict(dataset.val_X)
        val_true = np.argmax(dataset.val_Y, axis=0)
        val_acc = accuracy(val_preds, val_true)
        val_accuracies.append(val_acc)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        if epoch % print_every == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | Time: {epoch_time:.2f}s")

    # --- 测试集评估 ---
    test_preds = model.predict(dataset.test_X)
    test_true = np.argmax(dataset.test_Y, axis=0)
    test_acc = accuracy(test_preds, test_true)
    test_f1 = macro_f1_score(test_preds, test_true, dataset.num_classes)

    print(f"\n  [NumPy] 测试集准确率: {test_acc:.4f}")
    print(f"  [NumPy] 测试集 Macro F1: {test_f1:.4f}")

    # 打印混淆矩阵
    cm = confusion_matrix(test_preds, test_true, dataset.num_classes)
    print(f"\n  混淆矩阵:\n{cm}")

    return {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'epoch_times': epoch_times,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
    }


def plot_comparison(numpy_results, pytorch_results, config):
    """
    绘制双轨对比图表并保存到 notebooks/ 目录。

    生成：
    1. loss_comparison.png - Loss 收敛对比
    2. accuracy_evaluation.png - 验证集准确率对比
    """
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt

    figures_dir = config['output']['figures_dir']
    os.makedirs(figures_dir, exist_ok=True)

    epochs = range(1, len(numpy_results['train_losses']) + 1)
    dataset_name = config['data']['dataset']

    # 全局样式
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': '#f8f9fa',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'font.size': 12,
    })

    # ===== 图1: Loss 对比 =====
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, numpy_results['train_losses'],
            'o-', label='NumPy MLP', color='#2196F3', linewidth=2, markersize=4)
    ax.plot(epochs, pytorch_results['train_losses'],
            's-', label='PyTorch MLP', color='#FF5722', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Training Loss', fontsize=14)
    ax.set_title(f'Loss Convergence Comparison ({dataset_name})', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.set_xlim(1, len(epochs))
    fig.tight_layout()
    loss_path = os.path.join(figures_dir, 'loss_comparison.png')
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    print(f"\n[图表] Loss 对比已保存: {loss_path}")

    # ===== 图2: Accuracy 对比 =====
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, numpy_results['val_accuracies'],
            'o-', label='NumPy MLP', color='#4CAF50', linewidth=2, markersize=4)
    ax.plot(epochs, pytorch_results['val_accuracies'],
            's-', label='PyTorch MLP', color='#9C27B0', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Validation Accuracy', fontsize=14)
    ax.set_title(f'Accuracy Evaluation ({dataset_name})', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.set_xlim(1, len(epochs))
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    acc_path = os.path.join(figures_dir, 'accuracy_evaluation.png')
    fig.savefig(acc_path, dpi=150)
    plt.close(fig)
    print(f"[图表] Accuracy 对比已保存: {acc_path}")


def main():
    print("=" * 70)
    print("  MedMLP 双轨训练系统")
    print("=" * 70)

    # 加载配置
    config = load_config()
    dataset_name = config['data']['dataset']
    print(f"\n目标数据集: {dataset_name}")

    # 安装 pyyaml（如果尚未安装，此处配置已加载说明已安装）

    # ===== Track 1: NumPy MLP =====
    print("\n" + "=" * 70)
    print("  Track 1: 自研 NumPy MLP")
    print("=" * 70)
    numpy_results = train_numpy_mlp(config)

    # ===== Track 2: PyTorch Baseline =====
    print("\n" + "=" * 70)
    print("  Track 2: PyTorch Baseline")
    print("=" * 70)
    pytorch_results = train_pytorch_mlp(config)

    # ===== 对比总结 =====
    print("\n" + "=" * 70)
    print("  对比总结")
    print("=" * 70)
    print(f"  {'指标':<25} {'NumPy MLP':>12} {'PyTorch MLP':>12} {'差距':>10}")
    print(f"  {'-' * 60}")

    np_acc = numpy_results['test_accuracy']
    pt_acc = pytorch_results['test_accuracy']
    np_f1 = numpy_results['test_f1']
    pt_f1 = pytorch_results['test_f1']
    np_time = sum(numpy_results['epoch_times'])
    pt_time = sum(pytorch_results['epoch_times'])

    print(f"  {'测试准确率':<25} {np_acc:>12.4f} {pt_acc:>12.4f} {np_acc - pt_acc:>+10.4f}")
    print(f"  {'测试 Macro F1':<25} {np_f1:>12.4f} {pt_f1:>12.4f} {np_f1 - pt_f1:>+10.4f}")
    print(f"  {'总训练耗时 (s)':<25} {np_time:>12.2f} {pt_time:>12.2f} {np_time - pt_time:>+10.2f}")

    # ===== 绘制对比图表 =====
    plot_comparison(numpy_results, pytorch_results, config)

    print("\n" + "=" * 70)
    print("  训练完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
