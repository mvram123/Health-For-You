
# Importing Libraries
import pandas as pd
import numpy as np


class EDA():

    def __init__(self, path):
        self.data_path = path
        self.df = pd.read_csv(path)

    # Shape of dataset

    def shape(self):

        return self.df.shape

    # Numerical Features

    def numerical(self):
        data = self.df
        num_features = [i for i in data.columns if data[i].dtypes != 'O']

        return num_features

    # Categorical Features

    def categorical(self):
        data = self.df
        cat_features = [i for i in data.columns if data[i].dtypes == 'O']

        return cat_features

    # Checking Id Column

    def check_id(self):
        data = self.df
        for i in data.columns:
            if 'id' in i:
                return "Id column is there"
            else:
                return "No Id column"

    # Missing features in numerical features

    def num_missing_values(self):
        data = self.df

        num_missing = [i for i in data.columns if data[i].isnull().sum() > 0 and data[i].dtypes != 'O']

        return num_missing

    # Missing features in categorical features

    def cat_missing_values(self):
        data = self.df
        cat_missing = [i for i in data.columns if data[i].isnull().sum() > 0 and data[i].dtypes == 'O']

        return cat_missing

    # Highly Correlated features

    def high_corr_before_encoding(self):
        data = self.df
        cor_matrix = data.corr().abs()
        upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.7)]

        return to_drop








