"""
MedMLP PyTorch 对标基准
========================
使用 PyTorch nn.Sequential 搭建与自研 NumPy MLP 完全一致的网络架构，
读取相同的 .npz 数据文件，确保实验输入源完全对齐。
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def build_pytorch_mlp(input_dim, hidden_layers, num_classes):
    """
    构建与 NumPy 自研 MLP 层数、神经元数完全一致的 PyTorch 网络。
    
    Parameters
    ----------
    input_dim : int
    hidden_layers : list[int]
    num_classes : int
    
    Returns
    -------
    model : nn.Sequential
    """
    layers = []
    dims = [input_dim] + hidden_layers + [num_classes]
    
    for i in range(len(dims) - 1):
        linear = nn.Linear(dims[i], dims[i + 1])
        # 使用 He (Kaiming) 初始化，与 NumPy 版本对齐
        nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        
        # 最后一层不加 ReLU
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    
    return nn.Sequential(*layers)


def load_npz_for_pytorch(dataset_name, data_dir="data"):
    """
    加载 .npz 文件并转换为 PyTorch TensorDataset。
    
    注意：PyTorch 使用行向量模式 (N, D)，与 NumPy 自研版 (D, N) 不同。
    
    Returns
    -------
    train_dataset, val_dataset, test_dataset : TensorDataset
    input_dim : int
    num_classes : int
    """
    npz_path = os.path.join(data_dir, f"{dataset_name}.npz")
    data = np.load(npz_path)
    
    def process(images, labels):
        N = images.shape[0]
        X = images.reshape(N, -1).astype(np.float32) / 255.0  # (N, D)
        y = labels.flatten().astype(np.int64)
        return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    
    train_ds = process(data['train_images'], data['train_labels'])
    val_ds = process(data['val_images'], data['val_labels'])
    test_ds = process(data['test_images'], data['test_labels'])
    
    input_dim = train_ds.tensors[0].shape[1]
    num_classes = int(data['train_labels'].max()) + 1
    
    return train_ds, val_ds, test_ds, input_dim, num_classes


def train_pytorch_mlp(config):
    """
    训练 PyTorch MLP 基准模型。
    
    Parameters
    ----------
    config : dict
        从 YAML 加载的配置字典
        
    Returns
    -------
    results : dict
        包含 train_losses, val_accuracies, epoch_times, test_accuracy, test_f1
    """
    dataset_name = config['data']['dataset']
    data_dir = config['data']['data_dir']
    hidden_layers = config['model']['hidden_layers']
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    lr = config['training']['learning_rate']
    l2_lambda = config['training']['l2_lambda']
    print_every = config['output']['print_every']
    
    # 设备选择
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[PyTorch Baseline] 使用设备: {device}")
    
    # 数据加载
    train_ds, val_ds, test_ds, input_dim, num_classes = load_npz_for_pytorch(
        dataset_name, data_dir
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # 构建模型
    model = build_pytorch_mlp(input_dim, hidden_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_lambda)
    
    print(f"  网络结构: {input_dim} -> {' -> '.join(map(str, hidden_layers))} -> {num_classes}")
    
    # 训练循环
    train_losses = []
    val_accuracies = []
    epoch_times = []
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # --- 训练阶段 ---
        model.train()
        running_loss = 0.0
        num_batches = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
        
        avg_loss = running_loss / num_batches
        train_losses.append(avg_loss)
        
        # --- 验证阶段 ---
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        val_acc = correct / total
        val_accuracies.append(val_acc)
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        if epoch % print_every == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | Time: {epoch_time:.2f}s")
    
    # --- 测试集评估 ---
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    test_acc = np.mean(all_preds == all_labels)
    
    # 计算 Macro F1
    from src.metrics import macro_f1_score
    test_f1 = macro_f1_score(all_preds, all_labels, num_classes)
    
    print(f"\n  [PyTorch] 测试集准确率: {test_acc:.4f}")
    print(f"  [PyTorch] 测试集 Macro F1: {test_f1:.4f}")
    
    return {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'epoch_times': epoch_times,
        'test_accuracy': test_acc,
        'test_f1': test_f1,
    }
