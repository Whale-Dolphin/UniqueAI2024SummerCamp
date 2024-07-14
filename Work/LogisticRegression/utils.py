import numpy as np
import pandas as pd

class DataFrameImputer:
    def __init__(self, dataframe):
        self.dataframe = dataframe.copy()

    def mean_impute(self):
        df_filled = self.dataframe.copy()
        for column in df_filled.select_dtypes(include=[np.number]).columns:
            mean_value = np.round(df_filled[column].mean())
            df_filled[column].fillna(mean_value, inplace=True)
        return df_filled

    def mode_impute(self):
        df_filled = self.dataframe.copy()
        for column in df_filled.select_dtypes(include=[object]).columns:
            mode_value = df_filled[column].mode()[0]
            df_filled[column].fillna(mode_value, inplace=True)
        return df_filled

    def knn_impute(self, k=3):
        df_filled = self.dataframe.copy()
        for column in df_filled.columns:
            if df_filled[column].isnull().any():
                not_null = df_filled[~df_filled[column].isnull()]
                is_null = df_filled[df_filled[column].isnull()]
                for idx in is_null.index:
                    distances = np.linalg.norm(not_null.drop(columns=[column]).values - df_filled.loc[idx].drop(column).values, axis=1)
                    nearest_indices = not_null.index[np.argsort(distances)[:k]]
                    knn_value = np.round(not_null.loc[nearest_indices, column].mean())
                    df_filled.at[idx, column] = knn_value
        return df_filled


class DataFrameNormalizer:
    def __init__(self, dataframe):
        self.dataframe = dataframe.copy()

    def z_score_standardization(self):
        df_standardized = (self.dataframe - self.dataframe.mean()) / self.dataframe.std()
        return df_standardized

    def min_max_normalization(self):
        df_normalized = (self.dataframe - self.dataframe.min()) / (self.dataframe.max() - self.dataframe.min())
        return df_normalized

    def l2_normalization(self):
        df_l2_normalized = self.dataframe.apply(lambda x: x / np.sqrt(np.sum(x**2)), axis=1)
        return df_l2_normalized

    def max_abs_normalization(self):
        df_maxabs_normalized = self.dataframe / self.dataframe.abs().max()
        return df_maxabs_normalized


def ProcessDataframe(dataframe):
    dataframe = dataframe.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    dataframe['Sex'] = dataframe['Sex'].map({'female': 0, 'male': 1})
    dataframe['Embarked'] = dataframe['Embarked'].map({'S': 0, 'C': 1})
    return dataframe
