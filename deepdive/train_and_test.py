#train_and_test.py

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pandas as pd
from plotnine import *

import scaling02
import sampling_smote

import warnings
warnings.filterwarnings('ignore')


def train_and_test(model, x_trx, x_test, y_trx, y_test):
    model.fit(x_trx, y_trx)
    y_predict = model.predict(x_test)
    accuracy = round(accuracy_score(y_test, y_predict)*100, 2)
    print("Accuracy: ", accuracy, "%")
    return accuracy


def ggplot_point(x_trx, y_trx, x, y):
    data = pd.concat([x_trx, y_trx], axis=1)
    plot = (
        ggplot(data)
        + aes(x=x, y=y, fill='factor(Survived)')
        + geom_point()
    )
    print(plot)


if __name__ == '__main__':
    data, label = scaling02.fsdscaler()
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, shuffle=True)
    smoted_x, smoted_y = sampling_smote.smote(x_train, y_train)
    print("Original Data ")
    train_and_test(SVC(), x_train, x_test, y_train, y_test)
    print("Smoted Data ")
    train_and_test(SVC(), smoted_x, x_test, smoted_y, y_test)

    ggplot_point(x_train, y_train, x_train.columns[2], x_train.columns[3])
    ggplot_point(smoted_x, smoted_y, smoted_x.columns[2], smoted_x.columns[3])
