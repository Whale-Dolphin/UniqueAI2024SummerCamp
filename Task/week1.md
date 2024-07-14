# UniqueAI夏令营第一周任务

## 前置任务
1.配置linux环境，windows使用双系统或者使用wsl（建议使用wsl），mac用户可以基本跳过此步。如果你恰巧财力雄厚，我们当然也非常欢迎你直接使用linux服务器。

2.配置好git环境并在本仓库下提交第一个pr。

3.配置python环境，建议使用anaconda或者miniconda。

请使以在markdown中提交截图的方式提交该任务。

## 数据处理
数据处理是AI中基础且重要的一环，每个学习AI的同学应当首先学习数据处理的基本技巧。

数据集：[泰坦尼克号](https://uniquestudio.feishu.cn/drive/folder/fldcnV0PzAB5J8ZaoMp8WXho8if?from=from_copylink)

### 基础任务
1.缺失值处理，包括但不限于，knn填补，众数，均值填补，补零；

2.数据标准化，归一化;

3.对于离散型变量的OneHotEncoder。

### 进阶任务
使用torch中的Dataset和DataLoader类对上面处理过的数据集进行加载。

## 机器学习
可以用的py库：numpy, pandas, matplotlib，gym，time
禁止使用pytorch，tensorflow等深度学习库，禁用sklearn等直接调用模型的库。

### 基础任务
实现一个逻辑回归代码，数据集是上面的泰坦尼克数据集，要求通过给出的数据拟合是否生还。

### 进阶任务
1.尝试实现避免过拟合的方法；

2.尝试使用多种优化方法，包括但不限于SGD、Adam等；

3.尝试实现dropout。

import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# 1.分析总体存活率
csv_file_path = 'C:\\Users\\樊晨旭\\Desktop\\泰坦尼克号数据.csv'
csv_directory = os.path.dirname(csv_file_path)
os.chdir(csv_directory)
titanic = pd.read_csv(csv_file_path, encoding='GBK')  # 根据需要调整编码
sns.set_style('ticks')
plt.axis('equal')
titanic['Survived'].value_counts().plot.pie(autopct='%1.2f%%')
plt.show()
# 2.年龄数据的分布情况
sns.set()
sns.set_style('ticks')
# 对年龄缺失值进行补充
titanic_age = titanic[titanic['Age'].notnull()]
plt.figure(figsize=(12,5))
plt.subplot(121)
titanic_age['Age'].hist(bins=80)
plt.xlabel('Age')
plt.ylabel('Num')
plt.subplot(122)
titanic_age.boxplot(column='Age', showfliers= False)
titanic_age['Age'].describe()
plt.show()
# 3.性别和生存率
titanic[['Sex','Survived']].groupby('Sex').mean().plot.bar()
plt.show()
survived_sex = titanic.groupby(['Sex','Survived'])['Survived'].count()
print("女性存活率为:{:.2f}%".format(survived_sex.loc['female',1]/survived_sex.loc['female'].sum()*100))
print("男性存活率为:{:.2f}%".format(survived_sex.loc['male',1]/survived_sex.loc['male'].sum()*100))
# 4.年龄与存活率的关系
# 分析年龄和对应的船舱等级与存活率的关系
fig, ax = plt.subplots(1,2,figsize=(18,8))
sns.violinplot(x='Pclass', y='Age', hue='Survived', data=titanic_age, split=True, ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')

# 分析年龄和性别与存活率的关系
sns.violinplot(x='Sex', y='Age', hue='Survived', data=titanic_age, split=True, ax=ax[1])
ax[1].set_title('Sex and Age vs Survived')
plt.show()

# 计算老人和小孩的存活率
plt.figure(figsize=(18,4))
# 将年龄都转换成整数
titanic_age['Age_int']=titanic_age['Age'].astype(int)
average_age=titanic_age[['Age_int','Survived']].groupby('Age_int',as_index=False).mean()
sns.barplot(x='Age_int',y='Survived',data=average_age,palette='BuPu')
plt.grid(linestyle='--', alpha=0.5)
plt.show()
# 5.亲人与否以及亲人个数对存活率的影响

# 兄弟姐妹--sibsp
# 有无兄弟姐妹的影响
sibsp_df = titanic[titanic['SibSp'] != 0]
no_sibsp_df = titanic[titanic['SibSp'] == 0]
# 有无父母子女的影响
parch_df = titanic[titanic['Parch'] != 0]
no_parch_df = titanic[titanic['Parch'] == 0]

plt.figure(figsize=(12,3))
plt.subplot(141)
plt.axis('equal')
sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived','Survived'],autopct='%1.1f%%',colormap='Blues')

plt.subplot(142)
plt.axis('equal')
no_sibsp_df['Survived'].value_counts().plot.pie(labels=['No Survived','Survived'],autopct='%1.1f%%',colormap='Blues')

plt.subplot(143)
plt.axis('equal')
parch_df['Survived'].value_counts().plot.pie(labels=['No Survived','Survived'],autopct='%1.1f%%',colormap='Reds')

plt.subplot(144)
plt.axis('equal')
no_parch_df['Survived'].value_counts().plot.pie(labels=['No Survived','Survived'],autopct='%1.1f%%',colormap='Reds')

plt.show()

# 亲戚数量与存活率的关系
fig,ax = plt.subplots(1,2,figsize=(15,4))
titanic[['Parch','Survived']].groupby('Parch').mean().plot.bar(ax=ax[0])
titanic[['SibSp','Survived']].groupby('SibSp').mean().plot.bar(ax=ax[1])
# 整体家庭成员数量与存活率关系
titanic['family_size']=titanic['Parch']+titanic['SibSp']+1
titanic[['family_size','Survived']].groupby('family_size').mean().plot.bar(figsize=(15, 4))
plt.show()

# 6.票价与存活率的影响
fig,ax = plt.subplots(1,2, figsize=(15,4))
titanic['Fare'].hist(bins=70,ax=ax[0])
titanic.boxplot(column='Fare',by='Pclass',showfliers=False,ax=ax[1])
fare_not_survived=titanic['Fare'][titanic['Survived'] == 0]
fare_survived = titanic['Fare'][titanic['Survived'] ==1]
# 筛选数据
average_fare=pd.DataFrame([fare_not_survived.mean(),fare_survived.mean()])#均值
std_fare=pd.DataFrame([fare_not_survived.std(),fare_survived.std()])
average_fare.plot(std_fare,kind='bar',figsize=(15,4),grid=True)
plt.show()



# 读取数据集
df = pd.read_csv("C:\\Users\\樊晨旭\\Desktop\\泰坦尼克号数据.csv",encoding='GBK')

# 数据预处理
# 选择特征和目标变量
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = df['Survived']

# 处理缺失值
X['Age'].fillna(X['Age'].mean(), inplace=True)
X['Embarked'].fillna(X['Embarked'].mode()[0], inplace=True)

# 类别特征编码
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
X = pd.get_dummies(X, columns=['Embarked'], drop_first=True)

# 将数据转换为数组
X = np.array(X)
y = np.array(y)


# 定义逻辑回归模型类
class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iter=10000, fit_intercept=True):
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept

    # 定义sigmoid函数
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # 添加截距项
    def add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1) if self.fit_intercept else X

    # 计算损失函数
    def loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    # 训练模型
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.add_intercept(X)

        # 初始化权重
        self.theta = np.zeros(X.shape[1])

        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.learning_rate * gradient

    # 预测函数
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.add_intercept(X)

        return self.sigmoid(np.dot(X, self.theta))

    def predict(self, X, threshold=0.5):
        return self.predict_prob(X) >= threshold


# 划分训练集和测试集
def train_test_split(X, y, test_size=0.2):
    idx = int(len(X) * (1 - test_size))
    return X[:idx], X[idx:], y[:idx], y[idx:]
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 创建逻辑回归模型实例
model = LogisticRegression()

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 计算准确率
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy: {accuracy:.2f}')