# 使用numpy定义权重和偏置的初始化过程。

import numpy as np

# 定义MLP的参数

input_size = 5 # 输入特征数量
hidden_size = 50# 隐藏层大小
output_size = 2# 输出类别数量

# 初始化权重和偏置
W1 = np.random.randn(input_size, hidden_size) * 0.01  # 输入到隐藏层的权重
b1 = np.zeros((1, hidden_size))  # 隐藏层的偏置
W2 = np.random.randn(hidden_size, output_size) * 0.01  # 隐藏层到输出层的权重
b2 = np.zeros((1, output_size))  # 输出层的偏置

# 正向传播
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward_propagation(X, W1, b1, W2, b2):
    # 计算隐藏层的输出
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)

    # 计算输出层的输出
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    return A1, A2

# 损失函数与反向传播
def compute_loss(Y, A2):
    m = Y.shape[0]
    loss = -1 / m * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
    return loss


def backward_propagation(X, Y, A1, A2, W1, W2, b1, b2):
    m = Y.shape[0]

    # 输出层的梯度
    dZ2 = A2 - Y
    dW2 = 1 / m * np.dot(A1.T, dZ2)
    db2 = 1 / m * np.sum(dZ2, axis=0, keepdims=True)

    # 隐藏层的梯度
    dZ1 = np.dot(dZ2, W2.T) * A1 * (1 - A1)
    dW1 = 1 / m * np.dot(X.T, dZ1)
    db1 = 1 / m * np.sum(dZ1, axis=0, keepdims=True)

    return dW1, db1, dW2, db2

# 优化器的实现，使用梯度下降或其他优化算法来更新权重和偏置
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2
