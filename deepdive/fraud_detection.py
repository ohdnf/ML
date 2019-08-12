#fraud_detection.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
import seaborn as sns; sns.set()
import warnings
warnings.filterwarnings('ignore')


def pipeline(model, x_train, y_train, x_test, y_test, disc):
    model.fit(x_train, y_train.ravel())
    y_predict = model.predict(x_test)

    print(disc + "accuracy_score:  {:.2f}%".format(accuracy_score(y_test, y_predict) * 100))
    print(disc + "recall_score:    {:.2f}%".format(recall_score(y_test, y_predict) * 100))
    print(disc + "precision_score: {:.2f}%".format(precision_score(y_test, y_predict) * 100))
    print(disc + "roc_auc_score:   {:.2f}%".format(roc_auc_score(y_test, y_predict) * 100))

    print("=" * 64)

    cnf_matrix = confusion_matrix(y_test, y_predict)
    print(disc + "===>\n", cnf_matrix)
    print("Confusion Matrix Test[0,0]>=", cnf_matrix[0, 0])
    print("Confusion Matrix Test[0,1]>=", cnf_matrix[0, 1])
    print("Confusion Matrix Test[1,0]>=", cnf_matrix[1, 0])
    print("Confusion Matrix Test[1,1]>=", cnf_matrix[1, 1])

    print(disc + "matrix_accuracy_score: ", (cnf_matrix[1, 1] + cnf_matrix[0, 0]) /
          (cnf_matrix[1, 1] + cnf_matrix[1, 0] + cnf_matrix[0, 1] + cnf_matrix[0, 0]) * 100)
    print(disc + "matrix_recall_score:   ", (cnf_matrix[1, 1] /
                                             (cnf_matrix[1, 0] + cnf_matrix[1, 1]) * 100))
    print("\n")
    print("=" * 64)
    print("\n")


if __name__ == '__main__':
    data = pd.read_csv('./data/creditcard.csv')
    print(data.head())
    print(data.columns)
    print(pd.value_counts(data['Class']))

    pd.value_counts(data['Class']).plot.bar()
    plt.title('Fraud Class Histogram')
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.show()

    sdscaler = StandardScaler()
    data['normAmount'] = sdscaler.fit_transform(data['Amount'].values.reshape(-1, 1))
    data = data.drop(['Time', 'Amount'], axis=1)
    print(data.head())

    X = np.array(data.loc[:, data.columns != 'Class'])    # Attribute
    y = np.array(data.loc[:, data.columns == 'Class'])    # Class
    print('Shape of X: {}'.format(X.shape))
    print('Shape of y: {}'.format(y.shape))

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    print("Number transactions x_train dataset: ", x_train.shape)
    print("Number transactions x_train dataset: ", y_train.shape)
    print("Number transactions x_test dataset: ", x_test.shape)
    print("Number transactions x_test dataset: ", y_test.shape)

    print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
    print("Before OverSampling, counts of label '0': {}".format(sum(y_train==0)))
    print("y_train", y_train)
    print("y_train.ravel()", y_train.ravel())

    sm = SMOTE()
    x_train_res, y_train_res = sm.fit_sample(x_train, y_train.ravel())

    print('After OverSampling, the shape of x_train: {}'.format(x_train_res.shape))
    print('After OverSampling, the shape of y_train: {}'.format(y_train_res.shape))

    print("After OverSampling, counts of y_train '1': {}".format(sum(y_train_res==1)))
    print("After OverSampling, counts of y_train '0': {}".format(sum(y_train_res==0)))

    print('After OverSampling, the shape of x_test: {}'.format(x_test.shape))
    print('After OverSampling, the shape of y_test: {}'.format(y_test.shape))

    print("After OverSampling, counts of y_test '1': {}".format(sum(y_test==1)))
    print("After OverSampling, counts of y_test '0': {}".format(sum(y_test==0)))

    lr = LogisticRegression()
    rf = RandomForestClassifier()
    sv = SVC(gamma=0.001)

    pipeline(lr, x_train, y_train, x_test, y_test, "SMOTE 전 LogisticRegr")
    pipeline(lr, x_train_res, y_train_res, x_test, y_test, "SMOTE 후 LogisticRegr")
    pipeline(rf, x_train, y_train, x_test, y_test, "SMOTE 전 RandomForest")
    pipeline(rf, x_train_res, y_train_res, x_test, y_test, "SMOTE 후 RandomForest")
    pipeline(sv, x_train, y_train, x_test, y_test, "SMOTE 전 SupportVect")
    pipeline(sv, x_train_res, y_train_res, x_test, y_test, "SMOTE 후 SupportVect")
