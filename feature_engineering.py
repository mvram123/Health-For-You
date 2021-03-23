
# Importing Libraries

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from dataprep import EDA

lb = LabelEncoder()


class FeatEng():

    def __init__(self, path, target):
        self.df = pd.read_csv(path)
        self.eda = EDA(path=path)
        self.target = target

    # replacing numerical missing values with mean

    def rep_miss_with_mean(self):
        data = self.df
        missing = self.eda.num_missing_values()
        if len(missing) == 0:

            return data
        else:
            for i in missing:
                data[i] = data[i].fillna(data[i].mean())

            return data

    # replacing categorical missing values with unknown

    def rep_miss_with_unknown(self):
        data = self.rep_miss_with_mean()
        missing = self.eda.cat_missing_values()
        if len(missing) == 0:

            return data
        else:
            for i in missing:
                data[i] = data[i].fillna('Unknown')

            return data

    # dropping Id column if present

    def drop_id(self):
        data = self.rep_miss_with_unknown()
        id = self.eda.check_id()
        if id == "Id column is there":
            data.drop('id', axis=1, inplace=True)

            return data
        else:
            return data

    # dropping highly correlated features

    def drop_corr(self):
        data = self.drop_id()
        corr = self.eda.high_corr_before_encoding()
        if len(corr) == 0:
            # print("No correlated features")
            return data
        else:
            data.drop(corr,axis=1,inplace=True)

            return data

    # Encoding Categorical features

    def encoding(self):
        data = self.drop_corr()
        categorical = self.eda.categorical()
        if len(categorical) == 0:
            # print('No Categorical features')
            return data
        else:
            for i in categorical:
                data[i] = lb.fit_transform(data[i])

            return data

    # splitting into train and test datasets

    def splitting(self):
        data = self.encoding()
        target = self.target
        y = data[target]
        x = data.drop(target, axis=1)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

        return [x, y, x_train, x_test, y_train, y_test]

