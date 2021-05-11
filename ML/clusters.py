import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, y = make_blobs(centers=4, n_samples=200, random_state=0, cluster_std=0.7)

plt.scatter(X[:, 0], X[:, 1])
plt.show()

from sklearn.cluster import KMeans

model = KMeans(4)
model.fit(X)
print(model.cluster_centers_)
print(model.labels_)
plt.scatter(X[:, 0], X[:, 1], c=model.labels_)
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=100, color="red")
plt.show()
