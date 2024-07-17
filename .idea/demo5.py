import pandas as pd
import numpy as np

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
