#Exercise Set VI¶
#Exercise: Try clustering using the following algorithms.
# How do their results compare?
# Which performs the best? Tell a story why you think it performs the best.

# 1. Modules
import pandas as pd
import sklearn.cluster as skc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_samples

# Import data for transactions
transactions = pd.read_excel("./WineKMC.xlsx",  sheetname=1, names=["customer_name", "offer_id"])
# Import data for offers
offers = pd.read_excel("./WineKMC.xlsx", sheetname=0, names=["offer_id", "campaign", "varietal", "min_qty", "discount", "origin", "past_peak"])
# merge
df = pd.merge(offers,transactions)
# remove columnds we don't need
X = df[df.columns.difference(['customer_name','offer_id'])]
# encode categorical data and assign X to matrix of floats
X = pd.get_dummies(X, prefix=['cg', 'vr','or']).astype(float)

# Methods wich don't need K as an input
# 1. Affinity propagation
aprop = skc.AffinityPropagation().fit_predict(X)
# 4. DBSCAN
dbscan = skc.DBSCAN().fit_predict(X)


# Define number of clusters for those methdos that need it
K = 3
# 0. Kemeans
kmeans = skc.KMeans(n_clusters=K).fit_predict(X)

# 2. Spectral clustering
spclus =  skc.SpectralClustering(n_clusters=K).fit_predict(X)
# 3. Agglomerative clustering
aggclus =  skc.AgglomerativeClustering(n_clusters=K).fit_predict(X)

# Create data frame
cols = ['Kmeans', 'Spec', 'Agglo']
clusters = pd.DataFrame(
           np.vstack( (kmeans, spclus, aggclus)).T,
           columns =cols )

#clusters.Aff_Prop.unique().plot()
#len(np.unique(aprop))

# Functions to use in the jupyter notebook
def describe_no_K_needed():
    print(20*'==')
    print('Affinity Propagation found '+str(len(np.unique(aprop))) + ' clusters')
    print(20*'==')
    print('DBSCAN  found '+str(len(np.unique(dbscan))) + ' clusters')
    print(20*'==')
    
 
# 0. Kmeans (just for fun, I did not use OneHotLabeling before)
myK = np.arange(2,10); inertia = np.zeros_like(myK)
def elbow_OneHot():
    for idx, K in enumerate(myK):
        inertia[idx] = skc.KMeans(n_clusters=K).fit(X).inertia_
    plt.figure(100,(5,5))
    plt.plot(myK,inertia,'-o'); plt.xlabel('Clusters');plt.ylabel('Inertia');
    plt.title('Elbow Methods using OneHot endcoding for categorical variables')

def cluster_counts():
    cluster_count = pd.DataFrame()
    for cols in clusters.columns:
        cluster_count[cols] = clusters[cols].value_counts()
    plt.figure(100,(5,5)); cluster_count.T.plot.bar();
    plt.xlabel('Clustering Method'); plt.ylabel('Number of members in each cluster')
    