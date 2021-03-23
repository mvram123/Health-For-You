
# importing libraries

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from catboost import CatBoostClassifier
import pickle

from feature_engineering import FeatEng


class BestModel():

    def __init__(self, path, target):

        self.x_train = FeatEng(path=path, target=target).splitting()[2]
        self.x_test = FeatEng(path=path, target=target).splitting()[3]
        self.y_train = FeatEng(path=path, target=target).splitting()[4]
        self.y_test = FeatEng(path=path, target=target).splitting()[5]
        self.filename = (path.split('/')[1].split('.')[0] + '.pkl')

# Random Forest Algorithm

    def random_forest(self):

        try:

            # Model Training
            hyper_parameters = {'n_estimators': np.arange(100, 2100, 100),
                                'max_depth': np.arange(2, 21, 1),
                                'max_features': ['auto', 'sqrt', 'log2']}

            random = RandomForestClassifier()
            random_cv = RandomizedSearchCV(random, hyper_parameters, scoring='accuracy', n_jobs=-1, cv=5)
            random_cv.fit(self.x_train, self.y_train)

            # Model Prediction
            y_predict = random_cv.predict(self.x_test)
            a = accuracy_score(self.y_test, y_predict)

            return [a, random_cv]

        except Exception as e:
            print(e)

# Catboost Algorithm

    def cat_boost(self):

        try:
            # Model Training
            hyper_parameters = {'n_estimators': np.arange(100, 2600, 100)}

            cat = CatBoostClassifier(verbose=False)
            cat_cv = RandomizedSearchCV(cat, hyper_parameters, scoring='accuracy', n_jobs=-1, cv=5)
            cat_cv.fit(self.x_train, self.y_train)

            # Model Prediction
            y_predict1 = cat_cv.predict(self.x_test)

            a1 = accuracy_score(self.y_test, y_predict1)

            return [a1, cat_cv]

        except Exception as e:
            print(e)

# Best Model

    def model(self):

        if self.cat_boost()[0] > self.random_forest()[0]:
            model = self.cat_boost()[1]
            print('CatBoost chosen')

        else:

            model = self.random_forest()[1]
            print('RandomForest chosen')

        pickle.dump(model, open('Models/' + self.filename, 'wb'))


# Dataset Names with respect to Target variables

info = [['cancer.csv', 'diagnosis'], ['diabetes.csv', 'Outcome'], ['heart.csv', 'target'],
        ['kidney.csv', 'classification'], ['liver.csv', 'Dataset']]

# Creating Pickle File

for i in info:
    BestModel(path=('Data/' + i[0]), target=i[1]).model()




