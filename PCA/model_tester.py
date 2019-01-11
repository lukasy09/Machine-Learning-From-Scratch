from PCA import PCA
import matplotlib.pyplot as plt
import numpy as np

X = np.random.rand(1000, 3)

pca = PCA()
z=pca.fit_transform(data=X, k=2)
plt.scatter(z[:, 0], z[:, 1], s=40)
plt.xlabel("Z1")
plt.ylabel("Z2")
plt.title("Reduced dimension from 3D to 2D")
plt.show()
