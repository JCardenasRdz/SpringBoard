import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans
import matplotlib.cm as cm
import numpy as np

def mean_silhouette_scores(Xdata,range):
    scores = np.zeros((len(range),1))
    for idx,n_clusters in enumerate(range):
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(Xdata)
        scores[idx] = np.mean(silhouette_samples(Xdata, cluster_labels),axis=0)
    plt.plot(range,scores,'o--')
    plt.xlabel('Clusters')
    plt.ylabel('Mean Silhouette Scores ')
    plt.show()
    
 