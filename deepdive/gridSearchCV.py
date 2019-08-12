#gridSearchCV.py

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris


def grid_search_cv(model, x_data, y_data):
    parameters = {
        'C': [0.01, 0.1, 1, 10],
        'gamma': [0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2']
    }

    clf = GridSearchCV(model, parameters, cv=5, verbose=5, n_jobs=3)
    print("<<clf - fit>>")
    clf.fit(x_data, y_data)

    print("<<Best params>>", clf.best_params_, clf.best_estimator_, clf.best_score_)


if __name__ == '__main__':
    iris = load_iris()
    lr = LogisticRegression()

    gs_data = iris.data
    gs_label = iris.target
    gs_columns = iris.feature_names

    gs_data = pd.DataFrame(gs_data, columns=gs_columns)
    grid_search_cv(lr, gs_data, gs_label)

