import torch
from torch import nn

#定义训练模型
def train_model(net, train_iter, test_iter, loss, num_epochs, optimizer):
    """定义训练模型"""
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        net.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for X, y in train_iter:
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()

            epoch_loss += l.item()
            predicted = y_hat.round()#y四舍五入的值就是y的预测值
            total += y.size(0)
            correct += (predicted == y).sum().item()

        train_losses.append(epoch_loss / len(train_iter))
        train_accuracies.append(correct / total)

        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in test_iter:
                y_hat = net(X)
                predicted = y_hat.round()
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        test_accuracies.append(correct / total)
        print(f'epoch {epoch + 1}, loss {train_losses[-1]:.4f}, train acc {train_accuracies[-1]:.4f}, test acc {test_accuracies[-1]:.4f}')

    return train_losses, train_accuracies, test_accuracies

def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def train_model_CNN(net, train_iter, test_iter, loss, num_epochs, optimizer):
    """定义训练模型(CNN)"""
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        net.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for X, y in train_iter:
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()

            epoch_loss += l.item()
            total += y.size(0)
            correct += accuracy(y_hat, y)

        train_losses.append(epoch_loss / len(train_iter))
        train_accuracies.append(correct / total)

        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in test_iter:
                y_hat = net(X)
                total += y.size(0)
                correct += accuracy(y_hat, y)
        
        test_accuracies.append(correct / total)
        print(f'epoch {epoch + 1}, loss {train_losses[-1]:.4f}, train acc {train_accuracies[-1]:.4f}, test acc {test_accuracies[-1]:.4f}')

    return train_losses, train_accuracies, test_accuracies

def train_model_CNN_KL(net, train_iter, test_iter, loss, num_epochs, optimizer):
    """定义KL散度损失函数的训练模型(CNN)"""
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        net.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for X, y in train_iter:
            optimizer.zero_grad()
            y_hat = net(X)
            outputs = nn.functional.log_softmax(y_hat, dim=1)
            y_one_hot = torch.zeros(y.size(0), 10)
            y_one_hot[torch.arange(y.size(0)), y] = 1
            l = loss(outputs, y_one_hot)
            l.backward()
            optimizer.step()

            epoch_loss += l.item()
            total += y.size(0)
            correct += accuracy(y_hat, y)

        train_losses.append(epoch_loss / len(train_iter))
        train_accuracies.append(correct / total)

        net.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in test_iter:
                y_hat = net(X)
                total += y.size(0)
                correct += accuracy(y_hat, y)
        
        test_accuracies.append(correct / total)
        print(f'epoch {epoch + 1}, loss {train_losses[-1]:.4f}, train acc {train_accuracies[-1]:.4f}, test acc {test_accuracies[-1]:.4f}')

    return train_losses, train_accuracies, test_accuracies