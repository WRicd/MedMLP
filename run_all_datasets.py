"""
MedMLP 全数据集批量实验
========================
对 4 个 MedMNIST 数据集逐一运行 NumPy MLP vs PyTorch MLP 双轨训练，
为每个数据集生成独立的 Loss / Accuracy 对比图表，
并汇总所有实验结果用于 report.md 的生成。
"""

import os
import sys
import time
import json
import copy
import yaml
import numpy as np

from src.data_pipeline import MedMNISTDataset, get_batches
from src.layers import NumpyMLP
from src.optimizers import AdamOptimizer
from src.metrics import accuracy, macro_f1_score, confusion_matrix


def load_config(config_path="configs/base_config.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def train_numpy(dataset, config, epochs):
    """训练 NumPy MLP 并返回完整指标。"""
    hidden_layers = config['model']['hidden_layers']
    batch_size = config['training']['batch_size']
    lr = config['training']['learning_rate']
    l2_lambda = config['training']['l2_lambda']

    np.random.seed(42)
    model = NumpyMLP(dataset.input_dim, hidden_layers, dataset.num_classes)
    optimizer = AdamOptimizer(
        lr, beta1=config['adam']['beta1'],
        beta2=config['adam']['beta2'],
        epsilon=config['adam']['epsilon'],
    )

    train_losses, val_accuracies, epoch_times = [], [], []

    for epoch in range(epochs):
        t0 = time.time()
        eloss, nb = 0.0, 0
        for Xb, Yb in get_batches(dataset.train_X, dataset.train_Y, batch_size, shuffle=True):
            Z = model.forward(Xb)
            loss = model.compute_loss(Z, Yb, l2_lambda=l2_lambda)
            model.backward(Yb, l2_lambda=l2_lambda)
            for layer in model.dense_layers:
                optimizer.update(layer, layer.dW, layer.db)
            eloss += loss
            nb += 1

        avg_loss = eloss / nb
        train_losses.append(avg_loss)

        vp = model.predict(dataset.val_X)
        vt = np.argmax(dataset.val_Y, axis=0)
        va = accuracy(vp, vt)
        val_accuracies.append(va)

        et = time.time() - t0
        epoch_times.append(et)
        print(f"    [NumPy]   Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {va:.4f} | {et:.1f}s")

    # 测试
    tp = model.predict(dataset.test_X)
    tt = np.argmax(dataset.test_Y, axis=0)
    ta = accuracy(tp, tt)
    tf1 = macro_f1_score(tp, tt, dataset.num_classes)
    cm = confusion_matrix(tp, tt, dataset.num_classes)

    return {
        'train_losses': [float(x) for x in train_losses],
        'val_accuracies': [float(x) for x in val_accuracies],
        'epoch_times': epoch_times,
        'test_accuracy': float(ta),
        'test_f1': float(tf1),
        'confusion_matrix': cm.tolist(),
    }


def train_pytorch(dataset_name, config, epochs):
    """训练 PyTorch MLP 并返回完整指标。"""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from benchmarks.pytorch_mlp import build_pytorch_mlp, load_npz_for_pytorch

    data_dir = config['data']['data_dir']
    hidden_layers = config['model']['hidden_layers']
    batch_size = config['training']['batch_size']
    lr = config['training']['learning_rate']
    l2_lambda = config['training']['l2_lambda']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds, val_ds, test_ds, input_dim, num_classes = load_npz_for_pytorch(dataset_name, data_dir)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = build_pytorch_mlp(input_dim, hidden_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_lambda)

    train_losses, val_accuracies, epoch_times = [], [], []

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        rl, nb = 0.0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            rl += loss.item()
            nb += 1
        avg_loss = rl / nb
        train_losses.append(avg_loss)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out = model(Xb)
                _, pred = torch.max(out, 1)
                total += yb.size(0)
                correct += (pred == yb).sum().item()
        va = correct / total
        val_accuracies.append(va)

        et = time.time() - t0
        epoch_times.append(et)
        print(f"    [PyTorch] Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {va:.4f} | {et:.1f}s")

    # 测试
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            out = model(Xb)
            _, pred = torch.max(out, 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(yb.numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    ta = float(np.mean(all_preds == all_labels))
    tf1 = float(macro_f1_score(all_preds, all_labels, num_classes))
    cm = confusion_matrix(all_preds, all_labels, num_classes)

    return {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'epoch_times': epoch_times,
        'test_accuracy': ta,
        'test_f1': tf1,
        'confusion_matrix': cm.tolist(),
    }


def plot_dataset_charts(dataset_name, numpy_res, pytorch_res, epochs, figures_dir):
    """为单个数据集生成独立的 Loss 和 Accuracy 对比图。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    epoch_range = range(1, epochs + 1)

    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': '#f8f9fa',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'font.size': 12,
    })

    # Loss
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epoch_range, numpy_res['train_losses'],
            'o-', label='NumPy MLP', color='#2196F3', linewidth=2, markersize=4)
    ax.plot(epoch_range, pytorch_res['train_losses'],
            's-', label='PyTorch MLP', color='#FF5722', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Training Loss', fontsize=14)
    ax.set_title(f'Loss Convergence - {dataset_name}', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.set_xlim(1, epochs)
    fig.tight_layout()
    p = os.path.join(figures_dir, f'{dataset_name}_loss.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)

    # Accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epoch_range, numpy_res['val_accuracies'],
            'o-', label='NumPy MLP', color='#4CAF50', linewidth=2, markersize=4)
    ax.plot(epoch_range, pytorch_res['val_accuracies'],
            's-', label='PyTorch MLP', color='#9C27B0', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Validation Accuracy', fontsize=14)
    ax.set_title(f'Accuracy Evaluation - {dataset_name}', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.set_xlim(1, epochs)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    p = os.path.join(figures_dir, f'{dataset_name}_accuracy.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)


def plot_cross_dataset_summary(all_results, figures_dir):
    """绘制跨数据集汇总对比图。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    datasets = list(all_results.keys())
    np_acc = [all_results[d]['numpy']['test_accuracy'] for d in datasets]
    pt_acc = [all_results[d]['pytorch']['test_accuracy'] for d in datasets]
    np_f1 = [all_results[d]['numpy']['test_f1'] for d in datasets]
    pt_f1 = [all_results[d]['pytorch']['test_f1'] for d in datasets]

    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': '#f8f9fa',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'font.size': 12,
    })

    x = np.arange(len(datasets))
    width = 0.35

    # Accuracy 柱状图
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, np_acc, width, label='NumPy MLP', color='#2196F3', alpha=0.85)
    bars2 = ax.bar(x + width/2, pt_acc, width, label='PyTorch MLP', color='#FF5722', alpha=0.85)
    ax.set_xlabel('Dataset', fontsize=14)
    ax.set_ylabel('Test Accuracy', fontsize=14)
    ax.set_title('Cross-Dataset Accuracy Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.1)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'cross_dataset_accuracy.png'), dpi=150)
    plt.close(fig)

    # F1 柱状图
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, np_f1, width, label='NumPy MLP', color='#4CAF50', alpha=0.85)
    bars2 = ax.bar(x + width/2, pt_f1, width, label='PyTorch MLP', color='#9C27B0', alpha=0.85)
    ax.set_xlabel('Dataset', fontsize=14)
    ax.set_ylabel('Macro F1-Score', fontsize=14)
    ax.set_title('Cross-Dataset Macro F1 Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.1)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, 'cross_dataset_f1.png'), dpi=150)
    plt.close(fig)


def main():
    print("=" * 70)
    print("  MedMLP 全数据集批量实验")
    print("=" * 70)

    config = load_config()
    figures_dir = config['output']['figures_dir']
    os.makedirs(figures_dir, exist_ok=True)

    datasets_info = {
        'bloodmnist':  {'task': '外周血细胞分类', 'channels': 3, 'epochs': 20},
        'pathmnist':   {'task': '结直肠癌病理分类', 'channels': 3, 'epochs': 15},
        'octmnist':    {'task': '视网膜OCT分类',   'channels': 1, 'epochs': 15},
        'tissuemnist': {'task': '组织细胞多分类',   'channels': 1, 'epochs': 15},
    }

    all_results = {}

    for ds_name, ds_info in datasets_info.items():
        epochs = ds_info['epochs']
        print(f"\n{'=' * 70}")
        print(f"  数据集: {ds_name} ({ds_info['task']})")
        print(f"  Epochs: {epochs}")
        print(f"{'=' * 70}")

        # 加载数据
        dataset = MedMNISTDataset(ds_name, config['data']['data_dir'])

        # NumPy 轨道
        print(f"\n  --- NumPy MLP ---")
        np_res = train_numpy(dataset, config, epochs)

        # PyTorch 轨道
        print(f"\n  --- PyTorch MLP ---")
        pt_res = train_pytorch(ds_name, config, epochs)

        # 生成该数据集的独立图表
        plot_dataset_charts(ds_name, np_res, pt_res, epochs, figures_dir)
        print(f"\n  [图表] {ds_name}_loss.png, {ds_name}_accuracy.png 已保存")

        # 打印对比
        diff_acc = np_res['test_accuracy'] - pt_res['test_accuracy']
        diff_f1 = np_res['test_f1'] - pt_res['test_f1']
        np_time = sum(np_res['epoch_times'])
        pt_time = sum(pt_res['epoch_times'])

        print(f"\n  {'指标':<18} {'NumPy':>10} {'PyTorch':>10} {'差距':>10}")
        print(f"  {'-' * 50}")
        print(f"  {'测试准确率':<18} {np_res['test_accuracy']:>10.4f} {pt_res['test_accuracy']:>10.4f} {diff_acc:>+10.4f}")
        print(f"  {'Macro F1':<18} {np_res['test_f1']:>10.4f} {pt_res['test_f1']:>10.4f} {diff_f1:>+10.4f}")
        print(f"  {'耗时 (s)':<18} {np_time:>10.2f} {pt_time:>10.2f}")

        all_results[ds_name] = {
            'numpy': np_res,
            'pytorch': pt_res,
            'info': ds_info,
            'input_dim': dataset.input_dim,
            'num_classes': dataset.num_classes,
            'train_samples': dataset.train_X.shape[1],
            'val_samples': dataset.val_X.shape[1],
            'test_samples': dataset.test_X.shape[1],
        }

        # 释放内存
        del dataset

    # 汇总图
    print(f"\n{'=' * 70}")
    print(f"  生成跨数据集汇总图表")
    print(f"{'=' * 70}")
    plot_cross_dataset_summary(all_results, figures_dir)
    print(f"  cross_dataset_accuracy.png, cross_dataset_f1.png 已保存")

    # 保存原始结果为 JSON
    results_path = os.path.join(figures_dir, 'experiment_results.json')
    # 转换不可序列化的类型
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"  原始数据已保存: {results_path}")

    # 打印总表
    print(f"\n{'=' * 70}")
    print(f"  全数据集实验总表")
    print(f"{'=' * 70}")
    print(f"  {'数据集':<14} {'NumPy Acc':>10} {'PT Acc':>10} {'差距':>8} {'NumPy F1':>10} {'PT F1':>10}")
    print(f"  {'-' * 65}")
    for ds_name, r in all_results.items():
        na = r['numpy']['test_accuracy']
        pa = r['pytorch']['test_accuracy']
        nf = r['numpy']['test_f1']
        pf = r['pytorch']['test_f1']
        print(f"  {ds_name:<14} {na:>10.4f} {pa:>10.4f} {na-pa:>+8.4f} {nf:>10.4f} {pf:>10.4f}")

    print(f"\n  实验完成！")
    return all_results


if __name__ == '__main__':
    main()
