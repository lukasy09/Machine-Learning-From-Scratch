from PCA import PCA
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

X = np.random.rand(500, 3)
X = PCA.mean_normalize(X)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], cmap=plt.cm.viridis, linewidth=0.2)

print("XD")

pca = PCA()
z = pca.fit_transform(data=X, k=2)
plt.scatter(z[:, 0], z[:, 1], s=40)
plt.xlabel("Z1")
plt.ylabel("Z2")
plt.title("Reduced dimension from 3D to 2D")
plt.show()
