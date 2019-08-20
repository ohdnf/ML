#sampling_over_under.py

import pandas as pd
from scaler import importData
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def psampling(data, label):
    ros = RandomOverSampler()
    rus = RandomUnderSampler()
    oversampled_data, oversampled_label = ros.fit_resample(data, label)
    undersampled_data, undersampled_label = rus.fit_resample(data, label)
    oversampled_data = pd.DataFrame(oversampled_data, columns=data.columns)
    undersampled_data = pd.DataFrame(undersampled_data, columns=data.columns)

    print("Original Data Class Ratio\n{}".format(pd.get_dummies(label).sum()))
    print("\nOversampled Data Class Ratio\n{}".format(pd.get_dummies(oversampled_label).sum()))
    print("\nUndersampled Data Class Ratio\n{}".format(pd.get_dummies(undersampled_label).sum()))

    return oversampled_data, undersampled_data


if __name__ == '__main__':
    d, lbl = importData()
    psampling(d, lbl)
