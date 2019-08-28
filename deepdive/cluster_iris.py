from sklearn import datasets
import pandas as pd

# Data loading
iris = datasets.load_iris()

labels = pd.DataFrame(iris.target)
labels.columns = ['labels']
data = pd.DataFrame(iris.data)
data.columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
data = pd.concat([data, labels], axis=1)
# print(data.head())


# Extract feature
feature = data[['Sepal length', 'Sepal width']]
# print(feature.head())


# Create Model, Training & Prediction
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

model = KMeans(n_clusters=3, max_iter=300)
model.fit(feature)
predict = pd.DataFrame(model.predict(feature))
predict.columns = ['predict']
# concatenate labels to df as a new column
r = pd.concat([feature, predict], axis=1)
# print(r)

centers = pd.DataFrame(model.cluster_centers_, columns=['Sepal length', 'Sepal width'])
center_x = centers['Sepal length']
center_y = centers['Sepal width']

# Scatter plot
# plt.scatter(r['Sepal length'], r['Sepal width'], c=r['predict'], alpha=0.5)
# plt.scatter(center_x, center_y, s=50, marker='D', c='r')
# plt.show()


# Evaluate model w/ Cross tabulation
# ct = pd.crosstab(data['labels'], r['predict'])
# print(ct)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

scaler = StandardScaler()
model = KMeans(n_clusters=3)
pipeline = make_pipeline(scaler, model)
pipeline.fit(feature)
predict = pd.DataFrame(pipeline.predict(feature))
predict.columns = ['predict']
r = pd.concat([feature, predict], axis=1)
ct = pd.crosstab(data['labels'], r['predict'])
# print(ct)


# Feature distribution check
import matplotlib.pyplot as plt

# plt.subplot(1, 2, 1)
# plt.hist(data['Sepal length'])
# plt.title('Sepal length')
# plt.subplot(1, 2, 2)
# plt.hist(data['Sepal width'])
# plt.title('Sepal width')
# plt.show()


# Determine number of clusters w/ inertia value
ks = range(1, 10)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(feature)
    inertias.append(model.inertia_)

# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
