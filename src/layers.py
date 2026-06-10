"""
MedMLP 纯 NumPy 矩阵求导算子
==============================
严禁在此文件内导入 torch 或 scikit-learn。所有矩阵操作纯手工编写。
"""

import numpy as np


class DenseLayer:
    """
    全连接层（含 L2 正则化反向传播）。
    
    权重约定（列向量模式）：
    - W: (D_out, D_in)
    - b: (D_out, 1)
    - X: (D_in, N)
    - Z: (D_out, N) = W @ X + b
    """

    def __init__(self, input_dim, output_dim):
        # He (Kaiming) 初始化：W ~ N(0, sqrt(2/D_in))
        self.W = np.random.randn(output_dim, input_dim).astype(np.float32) * np.sqrt(
            2.0 / input_dim
        )
        self.b = np.zeros((output_dim, 1), dtype=np.float32)
        
        # 前向传播缓存
        self.X = None
        
        # 梯度缓存（供优化器使用）
        self.dW = None
        self.db = None

    def forward(self, X):
        """前向传播：Z = WX + b"""
        self.X = X
        Z = self.W @ X + self.b
        return Z

    def backward(self, dZ, l2_lambda=0.0):
        """
        反向传播：计算参数梯度并返回下游梯度。
        
        Parameters
        ----------
        dZ : ndarray, shape (D_out, N)
            上游梯度
        l2_lambda : float
            L2 正则化系数
            
        Returns
        -------
        dX : ndarray, shape (D_in, N)
            传给下游的梯度
        """
        N = self.X.shape[1]
        
        # dW = (1/N) * dZ @ X^T + (lambda/N) * W
        self.dW = (dZ @ self.X.T) / N + (l2_lambda / N) * self.W
        
        # db = (1/N) * sum(dZ, axis=1)，保持列向量
        self.db = np.sum(dZ, axis=1, keepdims=True) / N
        
        # dX = W^T @ dZ
        dX = self.W.T @ dZ
        
        return dX


class ReLULayer:
    """
    ReLU 激活函数层。
    
    前向：A = max(0, Z)
    反向：dZ = dA ⊙ I(Z > 0)
    """

    def __init__(self):
        self.Z = None

    def forward(self, Z):
        """前向传播：A = max(0, Z)"""
        self.Z = Z
        return np.maximum(0, Z)

    def backward(self, dA, l2_lambda=0.0):
        """反向传播：dZ = dA ⊙ I(Z > 0)"""
        return dA * (self.Z > 0).astype(np.float32)


class SoftmaxCrossEntropyLoss:
    """
    Softmax 与交叉熵联合算子（输出与损失一体化）。
    
    前向：计算 softmax 概率并返回交叉熵损失。
    反向：dZ = A - Y （经典简化公式）。
    """

    def __init__(self):
        self.probs = None

    def forward(self, Z, Y_true):
        """
        计算 Softmax + Cross-Entropy Loss。
        
        Parameters
        ----------
        Z : ndarray, shape (C, N)
            最后一层的线性输出（logits）
        Y_true : ndarray, shape (C, N)
            One-hot 真实标签
            
        Returns
        -------
        loss : float
            平均交叉熵损失
        """
        # 数值稳定性：Z = Z - max(Z, axis=0)
        Z_stable = Z - np.max(Z, axis=0, keepdims=True)
        exp_Z = np.exp(Z_stable)
        self.probs = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        
        N = Y_true.shape[1]
        # 交叉熵：L = -(1/N) * sum(Y * log(A))
        loss = -np.sum(Y_true * np.log(self.probs + 1e-15)) / N
        
        return loss

    def backward(self, Y_true):
        """反向传播：dZ = A - Y"""
        return self.probs - Y_true


class NumpyMLP:
    """
    模块化多层感知机，由 DenseLayer 和 ReLULayer 组装。
    
    网络结构示例（hidden_layers=[256, 128]）:
        Dense(D_in, 256) -> ReLU -> Dense(256, 128) -> ReLU -> Dense(128, C)
    """

    def __init__(self, input_dim, hidden_layers, num_classes):
        """
        Parameters
        ----------
        input_dim : int
            输入特征维度 D
        hidden_layers : list[int]
            隐藏层神经元数列表
        num_classes : int
            分类数 C
        """
        self.layers = []          # 所有层（Dense + ReLU 交替）
        self.dense_layers = []    # 仅全连接层（供优化器遍历）
        self.loss_fn = SoftmaxCrossEntropyLoss()
        
        # 构建隐藏层
        layer_dims = [input_dim] + hidden_layers + [num_classes]
        for i in range(len(layer_dims) - 1):
            dense = DenseLayer(layer_dims[i], layer_dims[i + 1])
            self.layers.append(dense)
            self.dense_layers.append(dense)
            
            # 最后一层不加 ReLU（由 Softmax 负责）
            if i < len(layer_dims) - 2:
                self.layers.append(ReLULayer())

    def forward(self, X):
        """前向传播：逐层计算"""
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def compute_loss(self, Z, Y_true, l2_lambda=0.0):
        """
        计算总损失 = 交叉熵 + L2 正则化项。
        
        Parameters
        ----------
        Z : ndarray
            最后一层的 logits
        Y_true : ndarray
            One-hot 标签
        l2_lambda : float
            L2 系数
            
        Returns
        -------
        loss : float
        """
        ce_loss = self.loss_fn.forward(Z, Y_true)
        
        # L2 正则化惩罚：(lambda / 2N) * sum(||W||^2)
        if l2_lambda > 0:
            N = Y_true.shape[1]
            l2_penalty = sum(
                np.sum(layer.W ** 2) for layer in self.dense_layers
            )
            ce_loss += (l2_lambda / (2 * N)) * l2_penalty
        
        return ce_loss

    def backward(self, Y_true, l2_lambda=0.0):
        """反向传播：从损失层反向逐层计算梯度"""
        dZ = self.loss_fn.backward(Y_true)
        
        for layer in reversed(self.layers):
            dZ = layer.backward(dZ, l2_lambda=l2_lambda)

    def predict(self, X):
        """预测：返回类别索引数组"""
        Z = self.forward(X)
        # Softmax（仅用于预测，不需要数值稳定版本）
        Z_stable = Z - np.max(Z, axis=0, keepdims=True)
        exp_Z = np.exp(Z_stable)
        probs = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        return np.argmax(probs, axis=0)
