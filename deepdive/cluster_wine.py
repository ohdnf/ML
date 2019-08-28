import os
from os.path import join
import copy
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine

wine = load_wine()
# print(wine.DESCR)

data = wine.data
label = wine.target
columns = wine.feature_names

data = pd.DataFrame(data, columns=columns)
print(data.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
data = pca.fit_transform(data)
# print(data.head())
print(data.shape)
# print(data.describe())
# print(data.info())


# Determine number of clusters with Inertia value
from sklearn.cluster import KMeans

# inertias = []
# for k in range(1, 10):
#     model = KMeans(n_clusters=k)
#     model.fit(data)
#     inertias.append(model.inertia_)
#
# plt.plot(range(1,10), inertias, '-o')
# plt.xlabel('number of clusters, k')
# plt.ylabel('inertia')
# plt.xticks(range(1, 10))
# plt.show()

# number of cluster will be 3
kmeans = KMeans(n_clusters=3)
kmeans.fit(data)
cluster = kmeans.predict(data)
plt.scatter(data[:, 0], data[:, 1], c=cluster, linewidths=1, edgecolors='black')
centers = pd.DataFrame(kmeans.cluster_centers_, columns=['x', 'y'])

plt.scatter(centers['x'], centers['y'], s=50, marker='D', c='r')
plt.show()
