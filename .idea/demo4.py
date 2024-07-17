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










