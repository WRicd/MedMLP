import numpy as np

class Layer:
    """全连接层"""
    def __init__(self, input_dim, output_dim):
        # 使用 He 初始化（适合 ReLU）
        self.W = np.random.randn(output_dim, input_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros((output_dim, 1))
        self.X = None
        self.Z = None
        self.dW = None
        self.db = None

    def forward(self, X):
        self.X = X
        self.Z = np.dot(self.W, X) + self.b
        return self.Z

    def backward(self, dZ, learning_rate):
        m = self.X.shape[1]
        self.dW = np.dot(dZ, self.X.T) / m
        self.db = np.sum(dZ, axis=1, keepdims=True) / m
        dX = np.dot(self.W.T, dZ)
        
        # 参数更新 (SGD)
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db
        return dX

class Activation:
    """ReLU 激活函数"""
    def __init__(self):
        self.Z = None

    def forward(self, Z):
        self.Z = Z
        return np.maximum(0, Z)

    def backward(self, dA):
        return dA * (self.Z > 0)

class CrossEntropyLoss:
    """Softmax 交叉熵损失"""
    def forward(self, Y_pred, Y_true):
        # 稳定性优化：减去最大值防止溢出
        exps = np.exp(Y_pred - np.max(Y_pred, axis=0, keepdims=True))
        self.probs = exps / np.sum(exps, axis=0, keepdims=True)
        m = Y_true.shape[1]
        loss = -np.sum(Y_true * np.log(self.probs + 1e-15)) / m
        return loss

    def backward(self, Y_true):
        return self.probs - Y_true

class NeuralNetwork:
    """模块化神经网络架构"""
    def __init__(self):
        self.layers = []
        self.loss_fn = CrossEntropyLoss()

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, Y_true, learning_rate):
        dZ = self.loss_fn.backward(Y_true)
        for layer in reversed(self.layers):
            dZ = layer.backward(dZ, learning_rate)

    def fit(self, X, Y, epochs, lr, batch_size):
        m = X.shape[1]
        loss_history = []
        for epoch in range(epochs):
            # 打乱数据
            permutation = np.random.permutation(m)
            X_shuffled = X[:, permutation]
            Y_shuffled = Y[:, permutation]
            
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[:, i:i+batch_size]
                Y_batch = Y_shuffled[:, i:i+batch_size]
                
                # 前向 -> 计算损失 -> 反向
                Y_pred = self.forward(X_batch)
                loss = self.loss_fn.forward(Y_pred, Y_batch)
                self.backward(Y_batch, lr)
                
            loss_history.append(loss)
            if epoch % (epochs // 10) == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}")
        return loss_history