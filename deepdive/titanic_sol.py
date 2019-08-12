#titanic_sol.py

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

train = pd.read_csv('../titanic/data/train.csv')
test = pd.read_csv('../titanic/data/test.csv')

# print("train.isnull().sum()", train.isnull().sum())
# train.isnull(): Age(177), Cabin(687), Embarked(2)
# print("test.isnull().sum()", test.isnull().sum())
# test.isnull(): Age(86), Cabin(327)

# Feature Engineering: Scaling, Sampling, Downsizing, Categorization
full_data = [train, test]

# Sex
for dataset in full_data:
    # dataset['Sex'] = dataset['Sex'].astype(str)
    dataset.loc[dataset['Sex'] == 'male', 'Sex'] = 0
    dataset.loc[dataset['Sex'] == 'female', 'Sex'] = 1

# Embarked
# train['Embarked'].value_counts(dropna=False)
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset['Embarked'] = dataset['Embarked'].astype(str)
# print("train.isnull().sum()", train.isnull().sum())
# isnull(): Age(177), Cabin(687), Embarked(0)

# Age
for dataset in full_data:
    dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
    dataset['Age'] = dataset['Age'].astype(int)
# train['AgeBand'] = pd.cut(train['Age'], 5)
# train['AgeBand'] = pd.qcut(train['Age'], 4)
# print(train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean())
for dataset in full_data:
    dataset.loc[dataset['Age'] <= 22, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 29), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 29) & (dataset['Age'] <= 35), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 35), 'Age'] = 3

# Family = Parch + SibSp
for dataset in full_data:
    dataset["Family"] = dataset["Parch"] + dataset["SibSp"]
    dataset["Family"] = dataset["Family"].astype(int)

# Title = Name
for dataset in full_data:
    dataset['Name'] = dataset['Name'].astype(str)
    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
                                                 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Others')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    dataset['Title'] = dataset['Title'].map({"Mr": 0, "Master": 1, "Miss": 2, "Mrs": 3, "Others": 4})

# Fare
# train['CategoricalFare'] = pd.qcut(train['Fare'], 4, duplicates='drop')
for dataset in full_data:
    # dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3


# Feature Selection
drop_elements = ['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked']
train = train.drop(drop_elements, axis=1)
test = test.drop(drop_elements, axis=1)

train_label = train['Survived']
train_data = train.drop('Survived', axis=1)

