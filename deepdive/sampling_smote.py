#sampling_smote.py

from imblearn.over_sampling import SMOTE
from scaler import importData
import pandas as pd


def smote(data, label):
    smote = SMOTE(k_neighbors=3)
    smoted_data, smoted_label = smote.fit_resample(data, label)
    smoted_data = pd.DataFrame(smoted_data, columns=data.columns)
    smoted_label = pd.DataFrame({'Survived': smoted_label})

    print("Original Data Class Ratio\n{}".format(pd.get_dummies(label).sum()))
    print("Smoted Data Class Ratio\n{}".format(pd.get_dummies(smoted_label).sum()))

    return smoted_data, smoted_label


if __name__ == '__main__':
    data, label = importData()
    smote(data, label)
