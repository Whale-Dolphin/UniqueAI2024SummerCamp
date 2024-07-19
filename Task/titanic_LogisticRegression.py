import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

# Step 1: 数据预处理
df=pd.read_csv("data.csv")
# 只有age，carbin，embark 数据缺失。
# age可能和结果有关，用平均值填补.embarked数量较少，使用众数填补.
df['Age']=df['Age'].fillna(df['Age'].mean())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])

##id,name和ticket对于每个人都不一样，可以不予考虑，连同carbin删掉
df.drop(columns='PassengerId',inplace=True)
df.drop(columns='Ticket',inplace=True)
df.drop(columns='Name',inplace=True)
df.drop(columns='Cabin',inplace=True)
#数据标准化
df['Pclass']=(df['Pclass']-df['Pclass'].min())/(df['Pclass'].max()-df['Pclass'].min())
df['Age']=(df['Age']-df['Age'].min())/(df['Age'].max()-df['Age'].min())
df['SibSp']=(df['SibSp']-df['SibSp'].min())/(df['SibSp'].max()-df['SibSp'].min())
df['Parch']=(df['Parch']-df['Parch'].min())/(df['Parch'].max()-df['Parch'].min())
df['Fare']=(df['Fare']-df['Fare'].min())/(df['Fare'].max()-df['Fare'].min())

#离散型变量的OneHotEncoder
encoder_sex=OneHotEncoder()
transformed_data=encoder_sex.fit_transform(df[['Sex']])
transformed_df = pd.DataFrame(transformed_data.toarray(), columns=encoder_sex.get_feature_names_out(['Sex']))
df = pd.concat([df, transformed_df], axis=1)
df.drop(columns=['Sex'], inplace=True)

encoder_embarked=OneHotEncoder()
transformed_data=encoder_sex.fit_transform(df[['Embarked']])
transformed_df = pd.DataFrame(transformed_data.toarray(), columns=encoder_sex.get_feature_names_out(['Embarked']))
df = pd.concat([df, transformed_df], axis=1)
df.drop(columns=['Embarked'], inplace=True)
# Step 2: 特征工程

# 提取特征和目标变量
X = df.drop('Survived', axis=1).values
y = df['Survived'].values

# 添加偏置项
X = np.insert(X, 0, 1, axis=1)  # 在第一列插入全为1的列作为偏置项

# Step 3: 逻辑回归模型

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    epsilon = 1e-5  # 防止log(0)的情况，导致计算出现错误
    cost = (-y * np.log(h + epsilon) - (1 - y) * np.log(1 - h + epsilon)).mean()
    return cost

def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []

    for i in range(iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history

# Step 4: 模型训练

# 初始化参数
theta = np.zeros(X.shape[1])
alpha = 0.01
iterations = 1000

# 训练模型
theta, cost_history = gradient_descent(X, y, theta, alpha, iterations)

# 打印训练后的参数
print(f"训练后的参数 theta：{theta}")

# Step 5: 模型评估
# 预测概率
def predict(X, theta):
    probabilities = sigmoid(X.dot(theta))
    return [1 if x >= 0.5 else 0 for x in probabilities]

y_pred = predict(X, theta)

# 计算准确率
accuracy = np.mean(y_pred == y)
print(f"模型准确率：{accuracy }")
