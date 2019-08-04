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

print(train.info())


# Feature Engineering

# SibSp & Parch
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# IsAlone: New Feature
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
# print (train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())

# Cabin
for dataset in full_data:
    dataset['Cabin'] = dataset['Cabin'].str[:1]

# Embarked
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
# print(train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean())

# Fare
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
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
cabin_mapping = {"A": 0, "B": 0.4, "C": 0.8, "D": 1.2, "E": 1.6, "F": 2, "G": 2.4, "T": 2.8}
title_mapping = {"Mr": 0, "Master": 1, "Miss": 2, "Mrs": 3, "Others": 4}

for dataset in full_data:
    # Sex
    dataset['Sex'] = dataset['Sex'].map(sex_mapping)
    # Cabin
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)
    # Title
    dataset['Title'] = dataset['Title'].map(title_mapping)
    # Embarked
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
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

train["Cabin"].fillna(train.groupby("Pclass")["Cabin"].transform("median"), inplace=True)
test["Cabin"].fillna(test.groupby("Pclass")["Cabin"].transform("median"), inplace=True)


# Feature Selection

drop_elements = ['PassengerId', 'Name', 'Ticket', 'SibSp', 'Parch', 'FamilySize']
train = train.drop(drop_elements, axis=1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis=1)
test = test.drop(drop_elements, axis=1)

train_data = train.drop('Survived', axis=1)
target = train['Survived']


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
    pipeline(clf, train_data, target, k_fold)

# Testing

# clf = SVC(probability=True)
clf = RandomForestClassifier(n_estimators=13)
clf.fit(train_data, target)

prediction = clf.predict(test)

submission = pd.DataFrame({
        "PassengerId": PassengerId,
        "Survived": prediction
    })

submission.to_csv('submission.csv', index=False)

submission = pd.read_csv('submission.csv')
print(submission.head())

