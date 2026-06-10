"""
MedMLP 数据流水线
==================
自适应加载 MedMNIST .npz 文件，支持列向量模式和高性能 Mini-batch 生成。
"""

import os
import numpy as np


class MedMNISTDataset:
    """
    通用 MedMNIST 数据集加载器。
    
    直接读取本地 .npz 文件，自适应扁平化并统一为列向量模式：
    - X: (D, N)  特征矩阵
    - Y: (C, N)  One-hot 标签矩阵
    """

    def __init__(self, dataset_name, data_dir="data"):
        """
        Parameters
        ----------
        dataset_name : str
            数据集名称，如 'bloodmnist', 'pathmnist', 'tissuemnist', 'octmnist'
        data_dir : str
            数据目录的相对路径
        """
        self.dataset_name = dataset_name
        npz_path = os.path.join(data_dir, f"{dataset_name}.npz")
        
        if not os.path.exists(npz_path):
            raise FileNotFoundError(
                f"数据文件 {npz_path} 不存在。请将 {dataset_name}.npz 放入 {data_dir}/ 目录。"
            )
        
        data = np.load(npz_path)
        
        # 加载并处理训练集、验证集、测试集
        self.train_X, self.train_Y, self.num_classes = self._process_split(
            data['train_images'], data['train_labels']
        )
        self.val_X, self.val_Y, _ = self._process_split(
            data['val_images'], data['val_labels'], num_classes=self.num_classes
        )
        self.test_X, self.test_Y, _ = self._process_split(
            data['test_images'], data['test_labels'], num_classes=self.num_classes
        )
        
        self.input_dim = self.train_X.shape[0]
        
        print(f"[数据加载完毕] {dataset_name}")
        print(f"  输入维度 D={self.input_dim}, 类别数 C={self.num_classes}")
        print(f"  训练集: {self.train_X.shape[1]} 样本")
        print(f"  验证集: {self.val_X.shape[1]} 样本")
        print(f"  测试集: {self.test_X.shape[1]} 样本")

    def _process_split(self, images, labels, num_classes=None):
        """
        对单个数据划分进行处理：扁平化 + 归一化 + One-hot 编码。
        
        Parameters
        ----------
        images : ndarray
            原始图像数组，形状 (N, 28, 28) 或 (N, 28, 28, 3)
        labels : ndarray
            标签数组，形状 (N, 1)
        num_classes : int or None
            类别数。若为 None，从标签推断。
            
        Returns
        -------
        X : ndarray, shape (D, N), float32
        Y : ndarray, shape (C, N), float32
        num_classes : int
        """
        N = images.shape[0]
        
        # 自适应扁平化 (Adaptive Flattening)
        # (N, 28, 28, 3) -> (N, 2352) 或 (N, 28, 28) -> (N, 784)
        X = images.reshape(N, -1).T  # (D, N)
        
        # 强制 float32 + 极差归一化
        X = X.astype(np.float32) / 255.0
        
        # One-hot 编码
        labels_flat = labels.flatten().astype(int)
        if num_classes is None:
            num_classes = int(labels_flat.max()) + 1
        
        Y = np.zeros((num_classes, N), dtype=np.float32)
        Y[labels_flat, np.arange(N)] = 1.0
        
        return X, Y, num_classes


def get_batches(X, Y, batch_size, shuffle=True):
    """
    高性能 Mini-batch 生成器。
    
    Parameters
    ----------
    X : ndarray, shape (D, N)
        特征矩阵（列向量模式）
    Y : ndarray, shape (C, N)
        One-hot 标签矩阵
    batch_size : int
        批大小
    shuffle : bool
        是否在每个 Epoch 开始前打乱数据
        
    Yields
    ------
    X_batch : ndarray, shape (D, batch_size)
    Y_batch : ndarray, shape (C, batch_size)
    """
    N = X.shape[1]
    
    if shuffle:
        perm = np.random.permutation(N)
        X = X[:, perm]
        Y = Y[:, perm]
    
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        yield X[:, start:end], Y[:, start:end]
