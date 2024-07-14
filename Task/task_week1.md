# 一、数据处理

- 第一步找出数据缺失项

![image-20240714221509401](https://github.com/captainwhite147/UniqueAI2024SummerCamp/blob/my_zhengyan/Task/images/image-20240714221509401.png)

可以看到只有age，cabin和embark有缺失数据，对于age常见的处理方法为均值填补，embarked缺失数据不多，而且只有三种情况，采用众数填补。数据集只有891项而cabin有687个缺项，直接删除。

在此题背景中id，name和ticket都是唯一的，为了后期数据的分析，直接删除

``

```python
# age可能和结果有关，用平均值填补.embarked数量较少，使用众数填补.
df['Age']=df['Age'].fillna(df['Age'].mean())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])

##id,name和ticket对于每个人都不一样，可以不予考虑，连同carbin删掉
df.drop(columns='PassengerId',inplace=True)
df.drop(columns='Ticket',inplace=True)
df.drop(columns='Name',inplace=True)
df.drop(columns='Cabin',inplace=True)

```

- 第二步数据的归一化

  这里只有`pclass，age，sibsp，parch`以数值形式出现，对四个变量min-max归一化处理

  ```python
  df['Pclass']=(df['Pclass']-df['Pclass'].min())/(df['Pclass'].max()-df['Pclass'].min())
  df['Age']=(df['Age']-df['Age'].min())/(df['Age'].max()-df['Age'].min())
  df['SibSp']=(df['SibSp']-df['SibSp'].min())/(df['SibSp'].max()-df['SibSp'].min())
  df['Parch']=(df['Parch']-df['Parch'].min())/(df['Parch'].max()-df['Parch'].min())
  df['Fare']=(df['Fare']-df['Fare'].min())/(df['Fare'].max()-df['Fare'].min())
  ```

- 第三步对于离散型变量的`OneHotEncoder`

  这里只剩下sex和embarked两个离散型变量，sex只可能有male和female两种情况，而embarked可能为从c,s和Q

  ```python
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
  ```

