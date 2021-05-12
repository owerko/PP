import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

mat = np.array([[-0.025125654, 0.018805442], [0.018805442, 0.119891771]])
print(f'Dana jest macierz Ex, Ey, Gxy: {mat}')
z = np.linalg.eigvals(mat)
print(f'Wartosci własne macierzy Ex, Ey, Gxy: {z}')
zv = np.linalg.eig(mat)
print(f'Wektory wlasne: {zv[1][0]} {zv[1][1]}')


Y = np.array([[0.0024, 0.018805442], [0.018805442, 0.1474]])
print(f'Zamcierz deformacji z wykorzystaniem 1 wektora własnego {Y}')

pca = PCA(2)
pca.fit(Y)

print("Wektory własne obliczone z użyciem PCA: Principal axes 1:", pca.components_[0])

