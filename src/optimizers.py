"""
MedMLP 工业级优化器
====================
支持 SGD with Momentum 和 Adam，统一接口设计。
"""

import numpy as np


class BaseOptimizer:
    """优化器基类，定义统一接口。"""

    def __init__(self, learning_rate):
        self.lr = learning_rate

    def update(self, layer, dW, db):
        """
        更新层参数。
        
        Parameters
        ----------
        layer : DenseLayer
            要更新的全连接层
        dW : ndarray
            权重梯度
        db : ndarray
            偏置梯度
        """
        raise NotImplementedError


class SGDMomentum(BaseOptimizer):
    """
    带动量的随机梯度下降。
    
    v_W = β * v_W + (1 - β) * dW
    W   = W - α * v_W
    """

    def __init__(self, learning_rate, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self._velocity = {}  # layer_id -> (v_W, v_b)

    def update(self, layer, dW, db):
        layer_id = id(layer)
        
        if layer_id not in self._velocity:
            self._velocity[layer_id] = (
                np.zeros_like(dW),
                np.zeros_like(db),
            )
        
        v_W, v_b = self._velocity[layer_id]
        
        v_W = self.momentum * v_W + (1 - self.momentum) * dW
        v_b = self.momentum * v_b + (1 - self.momentum) * db
        
        layer.W -= self.lr * v_W
        layer.b -= self.lr * v_b
        
        self._velocity[layer_id] = (v_W, v_b)


class AdamOptimizer(BaseOptimizer):
    """
    Adam 优化器：一阶矩、二阶矩 + 偏差修正。
    
    m = β₁ * m + (1 - β₁) * dW
    v = β₂ * v + (1 - β₂) * dW²
    m̂ = m / (1 - β₁ᵗ)
    v̂ = v / (1 - β₂ᵗ)
    W = W - α / (√v̂ + ε) * m̂
    """

    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self._moments = {}  # layer_id -> (m_W, v_W, m_b, v_b)
        self._t = {}        # layer_id -> timestep

    def update(self, layer, dW, db):
        layer_id = id(layer)
        
        if layer_id not in self._moments:
            self._moments[layer_id] = (
                np.zeros_like(dW),  # m_W
                np.zeros_like(dW),  # v_W
                np.zeros_like(db),  # m_b
                np.zeros_like(db),  # v_b
            )
            self._t[layer_id] = 0
        
        self._t[layer_id] += 1
        t = self._t[layer_id]
        
        m_W, v_W, m_b, v_b = self._moments[layer_id]
        
        # 一阶矩（均值）
        m_W = self.beta1 * m_W + (1 - self.beta1) * dW
        m_b = self.beta1 * m_b + (1 - self.beta1) * db
        
        # 二阶矩（未中心化方差）
        v_W = self.beta2 * v_W + (1 - self.beta2) * (dW ** 2)
        v_b = self.beta2 * v_b + (1 - self.beta2) * (db ** 2)
        
        # 偏差修正
        m_W_hat = m_W / (1 - self.beta1 ** t)
        m_b_hat = m_b / (1 - self.beta1 ** t)
        v_W_hat = v_W / (1 - self.beta2 ** t)
        v_b_hat = v_b / (1 - self.beta2 ** t)
        
        # 参数更新
        layer.W -= self.lr * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
        layer.b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
        
        self._moments[layer_id] = (m_W, v_W, m_b, v_b)
