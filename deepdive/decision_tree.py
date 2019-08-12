#decision_tree.py

# import logger
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


wine = load_wine()
data = wine.data
label = wine.target
columns = wine.feature_names
data = pd.DataFrame(data, columns=columns)

x_train, x_test, y_train, y_test = train_test_split(data, label, stratify=label)
tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)
score_tr = tree.score(x_train, y_train)
score_te = tree.score(x_test, y_test)
print('DT훈련 세트 정확도: {:.3f}'.format(score_tr))
print('DT테스트 세트 정확도: {:.3f}'.format(score_te))

tree1 = DecisionTreeClassifier(max_depth=2)
tree1.fit(x_train, y_train)
score_tr1 = tree1.score(x_train, y_train)
score_te1 = tree1.score(x_test, y_test)
print('DT훈련 depth 세트 정확도: {:.3f}'.format(score_tr1))
print('DT테스트 depth 세트 정확도: {:.3f}'.format(score_te1))

import graphviz
from sklearn.tree import export_graphviz

export_graphviz(tree1, out_file='tree1.dot',
                class_names=wine.target_names,
                feature_names=wine.feature_names,
                impurity=True,
                filled=True)

with open('tree1.dot') as file_reader:
    dot_graph = file_reader.read()

dot = graphviz.Source(dot_graph)
dot.render(filename='tree1')

# logger.debug("특성 중요도 첫번째:\n{}".format(tree.feature_importances_))

print('wine.data.shape=> ', wine.data.shape)
n_feature = wine.data.shape[1]
print(n_feature)
idx = np.arange(n_feature)
print('idx=> ', idx)
feature_imp = tree.feature_importances_
plt.barh(idx, feature_imp, align='center')
plt.yticks(idx, wine.feature_names)
plt.xlabel('feature importance', size=15)
plt.ylabel('feature', size=15)

plt.show()
