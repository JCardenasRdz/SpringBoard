# modules
from sklearn import datasets
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# import some data to play with
iris = datasets.load_iris()
X = iris.data

# Perform clustering for K =2 to K= 20
num_clusters = range(2,21)
inertia = np.zeros_like(num_clusters)
for idx,K in enumerate(num_clusters):
    inertia[idx] = KMeans(n_clusters=K, random_state=0).fit(X).inertia_

plt.plot(num_clusters,inertia,'o-'); plt.xlabel('Number of clusters'); plt.ylabel('Inertia')
plt.title('Iris Data Set')