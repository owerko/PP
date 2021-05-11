import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition

e = np.exp(1)
np.random.seed(4)


# pdf - probability density function

def pdf(x):
    return 0.5 * (stats.norm(scale=0.25 / e).pdf(x)
                  + stats.norm(scale=4 / e).pdf(x))


y = np.random.normal(scale=0.5, size=(3000))
x = np.random.normal(scale=0.5, size=(3000))
z = np.random.normal(scale=0.1, size=len(x))
d = np.random.normal(scale=1, size=len(x))

density = pdf(x) * pdf(y)
pdf_z = pdf(5 * z)

density *= pdf_z

a = x + y
b = 2 * y
c = a - b + z

norm = np.sqrt(a.var() + b.var())
a /= norm
b /= norm

X = np.column_stack((a, b, c, d))

print(X)

fig = plt.figure(1, figsize=(4, 3))
ax = Axes3D(fig, rect=(0, 0, .95, 1), elev=48, azim=134)
ax.scatter(X[:, 0], X[:, 1], X[:, 2])
ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()

pca = decomposition.PCA(n_components=2)
pca.fit(X)
Z = pca.transform(X)

#
print("Principal axes:", pca.components_)
print("2D Explained variance :", pca.explained_variance_)
print("2D Explained variance ratio:", pca.explained_variance_ratio_)
print("Mean:", pca.mean_)
#
Z = pca.transform(X)
plt.scatter(Z[:, 0], Z[:, 1])
plt.show()

pca = decomposition.PCA(n_components=1)
pca.fit(X)
Z = pca.transform(X)
print(pca.components_)
plt.axis('equal')
plt.scatter(Z[:, 0], np.zeros(len(Z[:, 0])))
plt.title('1D PCA')
plt.show()
