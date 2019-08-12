#cv_kfold.py

import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris


def train_test_(dataset):
    x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, random_state=0)
    logreg = LogisticRegression().fit(x_train, y_train)
    print("Train-Test Set Score: {:.2f}".format(logreg.score(x_test, y_test)))


def cross_valid(dataset):
    kf_data = dataset.data
    kf_label = dataset.target
    # kf_columns = dataset.feature_names

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
    scores = cross_validate(rf, kf_data, kf_label, cv=10, return_train_score=True)

    print("<< Score >>")
    print(scores)
    res_df = pd.DataFrame(scores)
    print("<< res_df >>")
    print(res_df)
    print("Average time and score:\n", res_df.mean())


def k_fold(dataset, label):
    kf = KFold(n_splits=5, random_state=0)
    for i, (train_idx, valid_idx) in enumerate(kf.split(dataset.values, label)):
        train_data, train_label = dataset.values[train_idx, :], label[train_idx]
        valid_data, valid_label = dataset.values[valid_idx, :], label[valid_idx]

        print("{} Fold train label\n{}".format(i, train_label))
        print("{} Fold valid label\n{}".format(i, valid_label))
        print("{} Fold, train_idx{}\ntrain_label\n{}".format(i, train_idx, train_label))


def stratified_k_fold(dataset, label):
    kf = StratifiedKFold(n_splits=5, random_state=0)
    for i, (train_idx, valid_idx) in enumerate(kf.split(dataset.values, label)):
        train_data, train_label = dataset.values[train_idx, :], label[train_idx]
        valid_data, valid_label = dataset.values[valid_idx, :], label[valid_idx]

        print("{} Fold train label\n{}".format(i, train_label))
        print("{} Fold valid label\n{}".format(i, valid_label))
        # print("Cross Validation Score: {:.2f}%".format(np.mean(val_scores)))


if __name__ == '__main__':
    iris = load_iris()
    iris_keys = iris.keys()
    # print(" << iris keys >> ")
    # print(iris_keys)

    kf_data = iris.data
    # print(kf_data)
    kf_label = iris.target
    kf_columns = iris.feature_names

    kf_data = pd.DataFrame(kf_data, columns=kf_columns)
    # print(kf_data)

    # train_test_(kf_data, kf_label)
    # cross_valid(kf_data, kf_label)

    print(" << kf_label >> ")
    print(kf_label)

    print(pd.value_counts(kf_label))
    print(kf_label.sum())
    print(kf_label.dtype)

    # k_fold(kf_data, kf_label)
    stratified_k_fold(kf_data, kf_label)

