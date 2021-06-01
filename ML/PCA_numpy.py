import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd


def PCA(X, num_components):
    # 1. Normalizacja zmiennej
    X_meaned = X - np.mean(X, axis=0)

    # 2. Oblicz macierz kowariancji
    cov_mat = np.cov(X_meaned, rowvar=False)

    # 3. Oblicz wartości własne i wektory własne
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

    # 4. Sortuj wartości własne w porządku malejącym
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]

    # 5. Wybierz podzbiór z uporządkowanej macierzy wartości własnych
    eigenvector_subset = sorted_eigenvectors[:, 0:num_components]

    # 6. Przekształć dane
    X_reduced = np.dot(eigenvector_subset.transpose(), X_meaned.transpose()).transpose()

    return X_reduced


# IRIS dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
data = pd.read_csv(url, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])


# prepare the data
x = data.iloc[:, 0:4]

# prepare the target
target = data.iloc[:, 4]

# Applying it to PCA function
mat_reduced = PCA(x, 2)

# Creating a Pandas DataFrame of reduced Dataset
principal_df = pd.DataFrame(mat_reduced, columns=['PC1', 'PC2'])

# Concat it with target variable to create a complete Dataset
principal_df = pd.concat([principal_df, pd.DataFrame(target)], axis=1)

plt.figure(figsize=(6, 6))
sb.scatterplot(data=principal_df, x='PC1', y='PC2', hue='target', s=60, palette='icefire')
plt.show()
