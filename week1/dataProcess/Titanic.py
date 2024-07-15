# 导入需要的一些库
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


def Standardization(data):
    """
    标准化函数:将一组数据转化为均值为0,标准差为1的标准正态分布
    """
    data = (data-data.mean())/data.std()
    return data
def Normolization(data):
    """
    归一化函数：将一组数据按比例缩放到(0,1)
    """
    data = (data-data.min())/(data.max()-data.min())
    return data

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


# 读取数据集，进行一些基本的分析
data_raw = pd.read_csv("week1\dataProcess\泰坦尼克号数据.csv")

##数据补全

# 我们通过补中位数的方式，对age列进行补全
data_raw['Age'].fillna(data_raw['Age'].median(),inplace=True)

# 我们通过补众数的方式来，对Embarked进行补全
data_raw['Embarked'] = data_raw['Embarked'].fillna(data_raw['Embarked'].mode().iloc[0])

# 由于cabin缺了大部分数据，所以我们直接用U代表Unkown对其进行填补
data_raw['Cabin'].fillna('U',inplace=True)

# 对Age和Fare的数据进行标准哈
data_raw['Age']  = Standardization(data_raw['Age'])
data_raw['Fare']  = Standardization(data_raw['Fare'])

##选取特征

# 标签
target = ['Survived'] 
y = data_raw[target]
# 用于预测的特征
fetures = ['Pclass','Sex','Age','Fare','Parch','SibSp','Embarked'] 

# 对特征进行编码
X = pd.get_dummies(data_raw[fetures],dtype=int)
# print(data_dummy.shape) (891,10)
# 一共891个样本
# print(x.columns)
# print(x.values[0],y.values[0])


## 制作数据集

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # 确保索引在范围内
        if idx >= len(self.data):
            raise IndexError("Index out of range")
        
        sample = torch.tensor(self.data.iloc[idx].values, dtype=torch.float32)  # 按行号索引数据并转换为 PyTorch 张量
        label = torch.tensor(self.labels.iloc[idx], dtype=torch.float32)  # 转换为 PyTorch 张量
        return sample, label
    
# 将已有数据集拆分成数据集与测试集
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
# print(X_train.shape) (713,10)
# print(X_test.shape)  (178,10)
# print(y_train.shape) (713,1)
# print(y_test.shape)  (178,1)

# 得到训练集与测试集
train_set = MyDataset(X_train,y_train)
test_set = MyDataset(X_test,y_test)

# 开始读取数据
batch_size = 4 # 一批读取多少个样本
# 构造训练和测试dataloader
train_dataloder = DataLoader(dataset=train_set,
                       batch_size=batch_size,
                       shuffle=True,
                       num_workers=0,
                       )
test_dataloader = DataLoader(dataset=test_set,
                       batch_size=batch_size,
                       shuffle=True,
                       num_workers=0,
                       )


# for batch_idx,(X,y) in enumerate(train_dataloder):
#     print('batch_idx',batch_idx)
#     print('Input:',X.shape) [4,10]
#     print('Output',y.shape) [4,1]

## 开始训练
epochs = 25 # 训练轮数
learning_rate = 0.01 # 学习率
train_losses = [] # 记录每一个batch的学习率
W_list = [] # 用于保存最后五个epoch的w
b_list = [] # 用于保存最后五个epoch的b


for epoch in range(epochs):
    loss_sum = 0 # 一个epoch总损失
    for batch_idx,(X,y) in enumerate(train_dataloder):
        
        batch_size = X.shape[0]
        feature_size = X.shape[1]
        # X的形状为 (batch_size,feature_size)
        
        # 随机初始化权重w
        if (epoch==0):
            W = np.zeros((feature_size,1))
            b = np.zeros((batch_size,1))
        # 计算y的预测值
        y_hat = 1/(1+np.exp(-(np.dot(X,W) + b)))
        
        # 计算交叉熵损失
        epsilon = 1e-8 # 加一个极小值，防止log(0)的出现
        loss = -y*np.log(y_hat+epsilon)-(1-y)*np.log(1-y_hat+epsilon)
        # 计入总损失中
        loss_sum+=loss.sum()

        # 计算关于权重 W 的梯度
        y = y.numpy() # 将y转换为numpy数组
        dW = np.dot(X.T, y_hat - y)
        # 计算关于偏置 b 的梯度
        db = np.sum(y_hat - y)

        #同时更新w和b的值
        W_temp = W - learning_rate*dW
        b_temp = b - learning_rate*db
        W = W_temp
        b = b_temp
        
        # 计算每个batch得到的平均损失
        batch_avg_loss = loss.mean()
        # print(f"Batch {batch_idx},batch_avg_loss={batch_avg_loss}")

    # 计算平均损失
    avg_loss = loss_sum/len(train_dataloder.dataset)
    train_losses.append(avg_loss)
    # print(f"Epoch {epoch}: Average Loss = {avg_loss}")
    
    # 根据学习曲线，当训练到最后五个eopch时，模型已经收敛
    if (epoch>=20):
        W_list.append(W)
        b_list.append(b)
        # print('W=',W)
        # print('b=',b)
# print(W_list[4],b_list[4])
# print(len(W_list))
# ## 可视化，制作学习曲线
# ****** 调试代码 *****
# plt.figure(figsize=(10, 6))
# plt.plot(range(1,len(train_losses)+1), train_losses, 'bo-')
# plt.title('TrainingLoss') # 整个图片的标题
# plt.xlabel('Epoch') # x轴为Batch的值
# plt.ylabel('Loss') # y轴为Loss的值
# plt.grid(True) # 绘制网格
# plt.show() 

## 测试部分

boundry = 0.6
times = 0  # 初始时间从0开始，因为我们要从第一个参数开始计算

for (W, b) in zip(W_list, b_list):
    correct_sum = 0  # 预测正确的总个数
    
    for batch_idx, (X, y) in enumerate(test_dataloader):
        # 计算y的预测值
        y_hat = 1 / (1 + np.exp(-(np.dot(X, W) + b)))
        # 分类
        y_hat_class = (y_hat > boundry).astype(int)
        # print(y_hat_class, y_hat)
        # 统计预测正确的个数
        correct = (y_hat_class == y.numpy()).sum()
        
        correct_sum += correct
    
    times += 1  # 更新时间变量，每次循环递增1
    
    # 计算准确率
    precision = correct_sum / len(test_dataloader.dataset)
    print(f"Parameter {times}: precision = {precision}")

"""
----- output ----- 
Parameter 1: precision = 0.7808988764044944
Parameter 2: precision = 0.7921348314606742
Parameter 3: precision = 0.8033707865168539
Parameter 4: precision = 0.8146067415730337
Parameter 5: precision = 0.8089887640449438
"""
   

    







