import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import model

# 设置超参数
batch_size = 256
lr = 0.9
num_epochs = 10

# 数据预处理和加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# print(next(iter(train_loader))[1])

# 定义卷积神经网络模型LeNet
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.Sigmoid(),
    nn.Linear(84, 10))
# 初始化模型、损失函数和优化器
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr)

#训练
train_losses, train_accuracies, test_accuracies = model.train_model_CNN(net, train_loader, test_loader, loss, num_epochs, optimizer)

#绘制训练过程的损失函数图像
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='train loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()

#绘制训练过程accuracy的变化图像
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label='train acc')
plt.plot(range(1, num_epochs + 1), test_accuracies, label='test acc', linestyle='--')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()

plt.show() #plt.show() 可能会阻塞主线程，等待用户关闭绘图窗口。

# 保存模型
torch.save(net.state_dict(), "mnist_cnn.pth")

