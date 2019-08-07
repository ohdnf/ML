#main.py

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

import warnings
warnings.filterwarnings('ignore')

# Importing Classifier Modules
# from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_val_score
# from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


# Data

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

PassengerId = test['PassengerId']

full_data = [train, test]


# Feature Engineering

# SibSp & Parch
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# IsAlone: New Feature
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
# print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

# Fare
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(0)

train['CategoricalFare'] = pd.qcut(train['Fare'], 4)

# Age
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)

train['CategoricalAge'] = pd.cut(train['Age'], 5)

# Title: New Feature
for dataset in full_data:
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Others')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


# Data Cleaning

sex_mapping = {"male": 0, "female": 1}
title_mapping = {"Mr": 0, "Master": 1, "Miss": 2, "Mrs": 3, "Others": 4}

for dataset in full_data:
    # Sex
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
    # Title
    dataset['Title'] = dataset['Title'].map(title_mapping)
    # Fare
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    # Age
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4


# Feature Selection

drop_elements = ['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked', 'FamilySize']
train = train.drop(drop_elements, axis=1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis=1)
test = test.drop(drop_elements, axis=1)

# train = pd.get_dummies(train)
# test = pd.get_dummies(test)

train_label = train['Survived']
train_data = train.drop('Survived', axis=1)


# Modelling

def pipeline(model, trx_data, label, cross_val):
    print(model.__class__.__name__)
    score = cross_val_score(model, trx_data, label, cv=cross_val, n_jobs=1, scoring='accuracy')
    print(score)
    print(round(np.mean(score) * 100, 2))
    print()
    acc[model.__class__.__name__] = round(np.mean(score) * 100, 2)


SEED = 0
classifiers = [
    KNeighborsClassifier(n_neighbors=5),
    SVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(n_estimators=13)
]
acc = {}

# Cross Validation(K-fold)
k_fold = KFold(n_splits=10, shuffle=True, random_state=SEED)

for clf in classifiers:
    pipeline(clf, train_data, train_label, k_fold)

# Testing

# clf = SVC(probability=True)
clf = RandomForestClassifier(n_estimators=13)
clf.fit(train_data, train_label)

prediction = clf.predict(test)

submission = pd.DataFrame({
        "PassengerId": PassengerId,
        "Survived": prediction
    })

submission.to_csv('submission.csv', index=False)

submission = pd.read_csv('submission.csv')
print(submission.head())

