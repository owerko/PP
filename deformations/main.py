from deformations.point import Point
from deformations.epoch import Epoch
import itertools as it
import numpy as np
import math
import os

# uzytkownik podaje indeksy naturalne od 1, ale obliczenia sa prowadzone jak w Pythonie -> indeksy od 0.

def ustawienia():
    pierwotna = input("Podaj nazwę pliku z epoką Pierowtną (name.txt): ")
    wtorna = input("Posaj nazwe pliku z epoka wtorna (name.txt): ")
    return pierwotna, wtorna


def helmert_points_idx():
    i1 = input('Podaj indeks pierwszego punktu referencyjnego: ')
    i2 = input('Podaj indeks drugiego punktu referencyjnego: ')
    return int(i1)-1, int(i2)-1


def A2d(plist, i1, i2):
    return np.array(
        [(float(plist[i1].x), -1 * float(plist[i1].y), 1, 0),
         (float(plist[i1].y), float(plist[i1].x), 0, 1),
         (float(plist[i2].x), -1 * float(plist[i2].y), 1, 0),
         (float(plist[i2].y), float(plist[i2].x), 0, 1)])

    # print(p.points[1].x)


def L2d(slist, i1, i2):
    L = np.array([float(slist[i1].x), float(slist[i1].y), float(slist[i2].x), float(slist[i2].y)])
    return L.transpose()


def lsf(A, L):
    At = A.transpose()
    AtA = np.matmul(At, A)
    AtA_1 = np.linalg.inv(AtA)
    AtL = np.matmul(At, L)
    return np.matmul(AtA_1, AtL)


def euklides(x1, x2, y1, y2):
    return math.sqrt((float(x2) - float(x1)) ** 2 + (float(y2) - float(y1)) ** 2)

def kombinacje():
    return it.combinations(range(6), 2)


if __name__ == "__main__":
    dane_pierowtne, dane_wtorne = ustawienia()
    with open(dane_pierowtne, 'r') as f:
        p = Epoch('primary')
        for line in f:
            name, x, y = line.split()
            point = Point(name, x, y)
            p.add_point(point)

    with open(dane_wtorne, 'r') as f:
        s = Epoch('secondary')
        for line in f:
            name, x, y = line.split()
            point = Point(name, x, y)
            s.add_point(point)
    print('Dane Epoki Pierwotnej: ')
    p.points_list()
    print('Dane Epoki Wtornej: ')
    s.points_list()

    i1, i2 = helmert_points_idx()

    print(A2d(p.points, i1, i2))
    print(L2d(s.points, i1, i2))

    A = A2d(p.points, i1, i2)
    L = L2d(s.points, i1, i2)
    X = lsf(A, L)

    print(f'{X[0]} {X[1]} {X[2]} {X[3]}')

    x_obl = []
    y_obl = []

    for point in p.points:
        x = X[0] * float(point.x) - X[1] * float(point.y) + X[2]
        y = X[1] * float(point.x) + X[0] * float(point.y) + X[3]
        print(x)
        print(y)
        x_obl.append(x)
        y_obl.append(y)

    for n, point in enumerate(s.points):
        d = 1000 * euklides(point.x, x_obl[n], point.y, y_obl[n])
        if d > 3:
            print(f'Punkt o indeksie {n+1} należy uznać za przemieszczony - d= {d:.3} mm')


