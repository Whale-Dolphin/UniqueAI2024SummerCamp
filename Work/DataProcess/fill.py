# 这里由于只有age和embarked有缺失值，所以只对这两列进行填补
# 故对其进行了四舍五入
# 对于其他值，需要根据实际判断是否需要四舍五入，或者采用其他取整方式

# 均值填补
def MeanFill(dataframe):
    df_filled = dataframe.copy()
    for column in df_filled.select_dtypes(include=[np.number]).columns:
        mean_value = np.round(df_filled[column].mean())
        df_filled[column].fillna(mean_value, inplace=True)
    return df_filled


# 众数填补
def ModeFill(dataframe):
    df_filled = dataframe.copy()
    for column in df_filled.select_dtypes(include=[np.object]).columns:
        mode_value = np.round(df_filled[column].mode()[0])
        df_filled[column].fillna(mode_value, inplace=True)
    return df_filled


# knn填补
def KnnImpute(df, k=3):
    df_filled = df.copy()
    for column in df.columns:
        if df[column].isnull().any():
            not_null = df[~df[column].isnull()]
            is_null = df[df[column].isnull()]
            for idx in is_null.index:
                distances = np.linalg.norm(not_null.drop(columns=[column]).values - df.loc[idx].drop(column).values, axis=1)
                nearest_indices = not_null.index[np.argsort(distances)[:k]]
                knn_value = np.round(not_null.loc[nearest_indices, column].mean())
                df_filled.at[idx, column] = knn_value
    return df_filled