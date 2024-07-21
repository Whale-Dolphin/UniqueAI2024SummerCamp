# 数据获取和预处理
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist  # 用于加载MNIST数据集

# 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 将像素值缩放到0到1之间
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 将图像数据从二维数组转为四维数组 (num_samples, height, width, channels)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# 将标签进行独热编码
num_classes = 10
y_train = np.eye(num_classes)[y_train]
y_test = np.eye(num_classes)[y_test]

# 划分验证集
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# 打印数据集的形状
print('训练集:', x_train.shape, y_train.shape)
print('验证集:', x_val.shape, y_val.shape)
print('测试集:', x_test.shape, y_test.shape)
# CNN的搭建
class ConvNet:
    def __init__(self):
        self.layers = [
            Conv2D(num_filters=16, filter_size=3, input_shape=(28, 28, 1), stride=1, padding='valid'),
            Activation('relu'),
            MaxPooling(pool_size=2, stride=2),
            Flatten(),
            Dense(128),
            Activation('relu'),
            Dense(10),
            Activation('softmax')
        ]
        self.loss_function = CrossEntropyLoss()
        self.optimizer = SGD(learning_rate=0.01)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def train(self, X, y):
        output = self.forward(X)
        loss = self.loss_function.forward(output, y)
        dloss = self.loss_function.backward()
        self.backward(dloss)
        self.optimizer.step()
        return loss

    def predict(self, X):
        logits = self.forward(X)
        return np.argmax(logits, axis=1)


# 定义卷积层、池化层、全连接层、激活函数、损失函数、优化器等类
class Conv2D:
    def __init__(self, num_filters, filter_size, input_shape, stride, padding):
        pass

    def forward(self, X):
        pass

    def backward(self, dout):
        pass

class MaxPooling:
    def __init__(self, pool_size, stride):
        pass

    def forward(self, X):
        pass

class Flatten:
    def forward(self, X):
        pass

class Dense:
    def __init__(self, units):
        pass

    def forward(self, X):
        pass

class Activation:
    def __init__(self, activation):
        pass

    def forward(self, X):
        pass

class CrossEntropyLoss:
    def forward(self, y_pred, y_true):
        pass

    def backward(self):
        pass

class SGD:
    def __init__(self, learning_rate):
        pass

    def step(self):
        pass
# 训练和评估
# 创建卷积神经网络模型实例
model = ConvNet()

# 训练模型
num_epochs = 10
batch_size = 64
num_batches = x_train.shape[0] // batch_size

for epoch in range(num_epochs):
    total_loss = 0
    for i in range(num_batches):
        batch_x = x_train[i * batch_size:(i + 1) * batch_size]
        batch_y = y_train[i * batch_size:(i + 1) * batch_size]

        loss = model.train(batch_x, batch_y)
        total_loss += loss

    # 每个epoch结束后在验证集上评估模型
    y_pred = model.predict(x_val)
    accuracy = np.mean(np.argmax(y_val, axis=1) == y_pred)

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / num_batches:.4f}, Accuracy: {accuracy:.4f}')

# 在测试集上评估模型
y_pred_test = model.predict(x_test)
test_accuracy = np.mean(np.argmax(y_test, axis=1) == y_pred_test)
print(f'测试集上的准确率: {test_accuracy:.4f}')
