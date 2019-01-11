import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt


class PCA:

    def __init__(self):
            self.data = None
            self.covariance_matrix = None

    @staticmethod
    def mean_normalize(data):
        (n_points, feature_number) = data.shape
        for feature_index in range(0, feature_number):
            mean = PCA.calculate_mi(data[:, feature_index])
            print(mean)
            for i in range(0, n_points):
                data[i, feature_index] -= mean
        return data

    @staticmethod
    def calculate_mi(vector):
        m = len(vector)
        numerator = 0
        for i in range(0, m):
            numerator += vector[i]
        return numerator / m

    def fit(self, data):
        self.data = data

    def compute_covariance_matrix(self):
        data = self.data
        (n_points, feature_number) = data.shape
        sigma = np.zeros([feature_number, feature_number], dtype=np.float64)
        for data_point in data:
            data_point = data_point.reshape(feature_number, 1)
            data_point_transposed = data_point.reshape(1, feature_number)
            product =  np.dot(data_point, data_point_transposed)
            sigma = np.add(sigma, product)
            
        return sigma / n_points



np.random.seed(0)
X, y = make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
#plt.show()

pca = PCA()
pca.fit(data=X)
pca.compute_covariance_matrix()


