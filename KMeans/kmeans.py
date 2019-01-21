from IndexedDataPoint import IndexedDataPoint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import random


class KMeans:

    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.data = None
        self.is_fitted = False

    def generate_random_centroid(self):
        data = self.data
        n_features = self.data.shape[1]

        centroid = []
        for feature_index in range(0, n_features):
            column = data[:, feature_index]
            minimal = np.min(column)
            maximal = np.max(column)

            random_coordinate = random.random() * (maximal - minimal) + minimal
            centroid.append(random_coordinate)

        return centroid



    def generate_centroids(self):
        for centroid in range(0, self.n_clusters):
            # self.current_centroids.append()
            pass


    def fit(self, X):

        self.data = X


np.random.seed(0)
X, y = make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
plt.show()

kmeans = KMeans(n_clusters= 2)
kmeans.fit(X = X)
kmeans.generate_random_centroid()