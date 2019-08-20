#bostonData.py

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd


boston = load_boston()
data = boston.data
label = boston.target
columns = boston.feature_names
data = pd.DataFrame(data, columns=columns)
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2)
