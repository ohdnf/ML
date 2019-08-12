#random_forest.py

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


digits = load_digits()
print(digits.keys())
print(digits['data'][0])
print(digits.images.shape)
print(digits['target_names'])

fig = plt.figure(figsize=(6, 6))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')

    ax.text(0, 7, str(digits.target[i]))

plt.show()

x_train_rf, x_test_rf, y_train_rf, y_test_rf = train_test_split(digits.data, digits.target)
rf_model = RandomForestClassifier(n_estimators=1000)
rf_model.fit(x_train_rf, y_train_rf)
y_pred_rf = rf_model.predict(x_test_rf)

print(metrics.classification_report(y_pred_rf, y_test_rf))
