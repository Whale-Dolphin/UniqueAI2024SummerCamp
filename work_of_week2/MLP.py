import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import model

# 加载和处理数据
data = pd.read_csv('../work_of_week1/data.csv')
X = data.drop('Survived', axis=1)
y = data['Survived']

X = pd.get_dummies(X, columns=['Sex', 'Embarked'])#进行独热编码
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)#划分训练集和测试集

X_train = X_train.copy()
X_test = X_test.copy()

#将数据标准化
scaler = StandardScaler()
numeric_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
X_train.loc[:, numeric_features] = scaler.fit_transform(X_train.loc[:, numeric_features])
X_test.loc[:, numeric_features] = scaler.transform(X_test.loc[:, numeric_features])

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

# 定义模型
net = nn.Sequential(nn.Linear(10,15),
                    nn.Tanh(),
                    nn.Linear(15,1),
                    nn.Sigmoid())

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)
net.apply(init_weights)

loss = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

batch_size = 64
#生成可迭代对象
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_epochs = 10
#训练
train_losses, train_accuracies, test_accuracies = model.train_model(net, train_loader, test_loader, loss, num_epochs, optimizer)

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

plt.show()
