# -*- coding: utf-8 -*-
"""missingvalue_prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1a-mzqMYk2rhMmF1C8yec4eNK6SK2FROP
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

iris = load_iris()

colnames = [col[:-5] for col in iris.feature_names]
colnames

iris_X = pd.DataFrame(iris.data, columns=colnames)
iris_Y = pd.DataFrame(iris.target, columns=['Species'])
iris_data = pd.concat([iris_X, iris_Y], axis=1)

iris_data.info()

iris_data.head()

data = iris_data.copy()

ctgr = data['petal length'] + data['petal width']
bins = (0, 2, 4, 6, 20)
group_names = ['S', 'X', 'L', 'XL']
categories = pd.cut(ctgr, bins, labels=group_names)
data['categorical_var'] = categories

sample = data.sample(n=10).index
data.iloc[sample, 5] = np.nan

data.head()

data.isna().sum()

data_ohc = pd.get_dummies(data)
sns.heatmap(data=data_ohc.corr(), annot=True, linewidths=.5, cmap='Blues')

x_train = data.dropna().drop(['categorical_var', 'Species'], axis=1)
y_train = data.dropna()['categorical_var']

x_test = data[data['categorical_var'].isna()].drop(['categorical_var', 'Species'], axis=1)

lr_param = {'C': np.logspace(-3, 3, 7), 'penalty': ['l1', 'l2']}
svm_param = {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1]}
knn_param = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']}

param_list = [lr_param, svm_param, knn_param]
model_list = [LogisticRegression(), SVC(), KNeighborsClassifier()]
grid_list = [GridSearchCV(MODEL, param_list[i], cv=3, n_jobs=4, verbose=True) for i, MODEL in enumerate(model_list)]

def voting_grid_mv(grid_list, x_train, y_train, x_test):
  pred_list=[]
  for i, model in enumerate(grid_list):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    pred_list += [y_pred]
  
  missing_value = pd.DataFrame(pred_list).mode().T
  missing_value.index = x_test.index

  return missing_value

mv = voting_grid_mv(grid_list, x_train, y_train, x_test)
mv
