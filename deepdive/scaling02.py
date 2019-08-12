#scaling02.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def importData():
    # data = pd.read_csv("../data/abalone.csv")
    data = pd.read_csv("../data/titanic_proc.csv")
    # data.shape
    # label = data['Sex']
    label = data['Survived']
    # del data['Sex']
    del data['Survived']
    return data, label


def mScaler():
    data, label = importData()
    mscaler = MinMaxScaler()
    mscaler.fit(data)
    mscaled_data = mscaler.transform(data)
    mscaled_data_f = pd.DataFrame(mscaled_data)
    print("min_values=> ", mscaled_data.min())
    print("max_values=> ", mscaled_data.max())
    print("data.mscaled_data_f()=> \n", mscaled_data_f.head())
    return mscaled_data_f, label


def stdScaler():
    data, label = importData()
    stdscaler = StandardScaler()
    stdscaler.fit(data)
    stdscaled_data = stdscaler.transform(data)
    stdscaled_pd = pd.DataFrame(stdscaled_data, columns=data.columns)
    print("stdscaled_data=>\n", stdscaled_pd.head())
    return stdscaled_pd, label


if __name__ == '__main__':
    mScaler()
    print("================")
    stdScaler()