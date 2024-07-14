import pandas as pd
from sklearn.preprocessing import OneHotEncoder
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
df.to_csv('processed_data.csv', index=False)