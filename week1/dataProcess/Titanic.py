# 导入需要的一些库
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # if torch.is_tensor(idx):
        #     idx = idx.tolist()
        # 确保索引在范围内
        if idx >= len(self.data):
            raise IndexError("Index out of range")
        
        sample = self.data.iloc[idx].values  # 按行号索引数据
        label = self.labels.iloc[idx].values 
        """
        此处data为dataFrame类型,labels为Series类型
        dataFrame和series类型经过切片之后均为series类型
        此处.values将他们转换为numpy数组类型
        """

        return sample, label

def Standardization(data):
    """
    标准化函数:将一组数据转化为均值为0,标准差为1的标准正态分布
    """
    return (data-data.mean())/data.std()
def Normolization(data):
    """
    归一化函数：将一组数据按比例缩放到(0,1)
    """
    return (data-data.min())/(data.max()-data.min())

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    将原有数据集拆分成数据集与测试集的函数
    X: 特征向量
    y: 标签值
    test_size: test数据集占整个数据集的比例
    randem_state: 随机种子
    """

    # 设置随机种子
    if random_state is not None:
        np.random.seed(random_state)
    
    # 确定测试集的大小（样本数）
    num_test = int(len(X) * test_size)
    
    # 生成随机索引
    indices = np.random.permutation(len(X))
    
    # 切分数据集
    X_train = X.iloc[indices[num_test:]]
    X_test = X.iloc[indices[:num_test]]
    y_train = y.iloc[indices[num_test:]]
    y_test = y.iloc[indices[:num_test]]
    
    return X_train, X_test, y_train, y_test

def data_preprocess(data_raw):
    """
    数据预处理函数：
        我们通过补中位数的方式,对age列进行补全
        我们通过补众数的方式来,对Embarked进行补全
        由于cabin缺了大部分数据,所以我们直接用U代表Unkown对其进行填补
    """
    # 数据补全与特征缩放
    data_raw['Age'].fillna(data_raw['Age'].median(),inplace=True)
    data_raw['Embarked'] = data_raw['Embarked'].fillna(data_raw['Embarked'].mode().iloc[0]) 
    data_raw['Cabin'].fillna('U',inplace=True)
    data_raw['Age']  = Standardization(data_raw['Age'])
    data_raw['Fare']  = Standardization(data_raw['Fare'])

    # 标签
    target = ['Survived'] 
    y = data_raw[target]
    # 用于预测的特征
    fetures = ['Pclass','Sex','Age','Fare','Parch','SibSp','Embarked'] 
    X = pd.get_dummies(data_raw[fetures],dtype=int)
    
    return X,y
    # print(data_dummy.shape) (891,10)
    # 一共891个样本
    # print(x.columns)
    # print(x.values[0],y.values[0])

def train_model(train_dataloader, epochs=55,lr=0.0001,lambda_reg = 0):
    """模型训练函数"""
    train_losses = [] # 储存每个epoch的loss
    W_list, b_list = [], [] # 存储最后五个权重的列表

    # 初始化 W 和 b
    feature_size = next(iter(train_dataloader))[0].shape[1]
    W = np.zeros((feature_size, 1))
    b = 0
    
    for epoch in range(epochs):
        loss_sum = 0
        for batch_idx, (X, y) in enumerate(train_dataloader):
            batch_size = X.shape[0]
            
            y_hat = 1 / (1 + np.exp(-(np.dot(X, W) + b)))
            epsilon = 1e-8
            regulation = lambda_reg * ((W * W).sum()) # 正则项
            loss = -y * np.log(y_hat + epsilon) - (1 - y) * np.log(1 - y_hat + epsilon) + regulation
            regulation_loss = loss.sum() + regulation
            loss_sum += regulation_loss
            
            y = y.numpy()
            dW = (np.dot(X.T, y_hat - y) ) / batch_size + 2 * lambda_reg * W
            db = np.sum(y_hat - y) / batch_size
            # print("dW = ",dW)
            W -= lr * dW
            b -= lr * db
            
        avg_loss = loss_sum / (len(train_dataloader.dataset) * 2) 
        train_losses.append(avg_loss)
        
        if epoch > epochs-6:
            W_list.append(W.copy())
            b_list.append(b)
    
    return train_losses, W_list, b_list

def evaluate_model(test_dataloader, W_list, b_list, boundary=0.6):
    """模型评估函数"""
    for idx,(W, b) in enumerate(zip(W_list, b_list)):
        correct_sum = 0
        
        for batch_idx, (X, y) in enumerate(test_dataloader):
            y_hat = 1 / (1 + np.exp(-(np.dot(X, W) + b)))
            y_hat_class = (y_hat > boundary).astype(int)
            correct = (y_hat_class == y.numpy()).sum()
            correct_sum += correct
        
        acc = (correct_sum / len(test_dataloader.dataset))*100
        print(f"Parameter {idx + 1}: acc = {acc:.2f}%")


# 主程序

# 读取数据集
data_raw = pd.read_csv("week1\dataProcess\泰坦尼克号数据.csv")

# 数据预处理
X, y = data_preprocess(data_raw)

# 将数据集拆分成训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# 制作数据集
train_set = MyDataset(X_train, y_train)
test_set = MyDataset(X_test, y_test)

# 构造dataLoader
batch_size = 4
train_dataloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataloader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, num_workers=0)

# 训练
train_losses, W_list, b_list = train_model(train_dataloader,epochs=200, lr=0.005,lambda_reg=0)

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses) + 1), train_losses, 'bo-')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

for idx,(W,b) in enumerate(zip(W_list,b_list)):
    print(f'{idx}: ',W,b)

# 测试
evaluate_model(test_dataloader, W_list, b_list)

   

    







