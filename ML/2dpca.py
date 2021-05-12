import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def arrow(v1, v2, ax):
    arrowprops = dict(arrowstyle='->',
                      linewidth=2,
                      shrinkA=0, shrinkB=0)
    ax.annotate("", v2, v1, arrowprops=arrowprops)


n = 50
x = np.arange(-n / 2, n / 2, 1, dtype=np.float64)
m = np.random.uniform(0.6, 1, (n,))
b = np.random.uniform(5, 15, (n,))
y = m * x + b

X = np.column_stack((x, y))

plt.scatter(X[:, 0], X[:, 1])
plt.show()
#
pca = PCA(2)
pca.fit(X)
#
print("Principal axes:", pca.components_)
print("2D Explained variance :", pca.explained_variance_)
print("2D Explained variance ratio:", pca.explained_variance_ratio_)
print("Mean:", pca.mean_)
#
X2D = pca.transform(X)
plt.scatter(X2D[:, 0], X2D[:, 1])
plt.show()

k = 20

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].axis('equal')
axes[0].scatter(X[:, 0], X[:, 1])
axes[1].axis('equal')
axes[1].set_xlim(-30, 30)
axes[1].scatter(X2D[:, 0], X2D[:, 1])
axes[0].arrow(0, 10, pca.components_[0][0] * k, pca.components_[0][1] * k, head_width=0.05 * k, head_length=0.1 * k,
              fc='k', ec='k')
axes[0].arrow(0, 10, pca.components_[1][0] * k, pca.components_[1][1] * k, head_width=0.05 * k, head_length=0.1 * k,
              fc='k', ec='k')
axes[1].arrow(0, 0, np.linalg.norm(pca.components_[0]) * k, 0, head_width=0.05 * k, head_length=0.1 * k, fc='k', ec='k')
axes[1].arrow(0, 0, 0, np.linalg.norm(pca.components_[0]) * k, head_width=0.05 * k, head_length=0.1 * k, fc='k', ec='k')
axes[0].set_title("Original")
axes[1].set_title("Transformed")
plt.show()

#1D PCA

pca = PCA(n_components=1)
pca.fit(X)
X1D = pca.transform(X)
print(pca.components_)
plt.axis('equal')
plt.scatter(X2D[:, 0], np.zeros(len(X1D[:, 0])))
plt.title('1D PCA')
plt.show()
