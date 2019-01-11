import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt


class PCA:

    def __init__(self):
            self.data = None
            self.covariance_matrix = None
            self.reduced_components = None
            self.sigma = None

    @staticmethod
    def mean_normalize(data):
        (n_points, feature_number) = data.shape
        for feature_index in range(0, feature_number):
            mean = PCA.calculate_mi(data[:, feature_index])
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

    def fit_transform(self, data, k=1):
        self.data = data
        self.reduced_components = k

        covariance_matrix = self.compute_covariance_matrix()
        u_matrix = self.compute_usv(sigma=covariance_matrix)
        z_vector = self.reduce(u_matrix=u_matrix)

        return z_vector

    def compute_covariance_matrix(self):
        data = self.data
        (n_points, feature_number) = data.shape
        sigma = np.zeros([feature_number, feature_number], dtype=np.float64)
        for data_point in data:
            data_point = data_point.reshape(feature_number, 1)
            data_point_transposed = data_point.reshape(1, feature_number)
            product = np.dot(data_point, data_point_transposed)
            sigma = np.add(sigma, product)
        return sigma / n_points

    @staticmethod
    def compute_usv(sigma):
        u_matrix, _, _ = np.linalg.svd(sigma, full_matrices=True)
        return u_matrix

    def reduce(self, u_matrix):
        data = self.data
        n_points, n_features = self.data.shape
        k = self.reduced_components
        z_vector = np.zeros([n_points, k], dtype=np.float64)
        u_reduced = u_matrix[:, 0:k]
        u_reduced_transposed = np.transpose(u_reduced)

        for i in range(0, n_points):
            z_vector[i] = np.dot(u_reduced_transposed, data[i].reshape(2, 1))
        print(z_vector)
        return z_vector

np.random.seed(0)
X = np.array([[[1,2,3],[4,5,3]], [[3,4,5], [7,8,9]]])
#plt.show()

print(X.shape)
pca = PCA()
z = pca.fit_transform(data=X, k=2)


