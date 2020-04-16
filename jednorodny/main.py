from jednorodny.point import Point
from jednorodny.epoch import Epoch
import math
import numpy as np


def radius(perimeter):
    return perimeter / (2 * math.pi)


def azimuth(radius, perimeter):
    return perimeter * 400 / (2 * math.pi * radius)


def A(epoch):
    lst = []
    for i in range(len(epoch.points)):
        lst.append((1, epoch.points[i].y, -1 * epoch.points[i].x))
    return np.array(lst)


def L(epoch):
    lst = []
    for i in range(len(epoch.points)):
        lst.append(float(epoch.points[i].dz))
    return np.array(lst)

def Ahz(epoch):
    lst = []
    for i in range(len(epoch.points)):
        lst.append((1, 0, epoch.points[i].x, 0, epoch.points[i].y, -1*epoch.points[i].y))
    for i in range(len(epoch.points)):
        lst.append((0, 1, 0, epoch.points[i].y, epoch.points[i].x, epoch.points[i].x))
    return np.array(lst)

def Lhz(epoch):
    lst = []
    for i in range(len(epoch.points)):
        lst.append(float(epoch.points[i].dx))
    for i in range(len(epoch.points)):
        lst.append(float(epoch.points[i].dy))
    return np.array(lst)

def lsf(A, L):
    At = A.transpose()
    AtA = np.matmul(At, A)
    AtA_1 = np.linalg.inv(AtA)
    AtL = np.matmul(At, L)
    return np.matmul(AtA_1, AtL)


if __name__ == '__main__':
    perimeter = float(input('Podaj obwod fundamentu: '))
    r = radius(perimeter)
    with open('data.txt', 'r') as f:
        epoka = Epoch('pomiar')
        for line in f:
            nr, pm, dx, dy, dz = line.split()
            az = azimuth(r, float(pm))
            x = r * math.cos(az * math.pi / 200)
            y = r * math.sin(az * math.pi / 200)
            point = Point(nr, pm, dx, dy, dz, x, y)
            epoka.add_point(point)
            # dim += 1

    print(f'Macierz A ma postać:')
    print(A(epoka))
    print(f'Macierz L ma postać:')
    print(L(epoka))
    X = lsf(A(epoka), L(epoka))
    print(f'Osiadanie wynosi {X[0]:.2f} mm, obrot wokol osi X {X[1]:.3f} mm/m, obrot wokol osi Y {X[2]:.3f} mm/m')
    Xhz = lsf(Ahz(epoka), Lhz(epoka))
    print(f'Przemieszczenie X {Xhz[0]:.2f} mm, Przemieszczenie Y {Xhz[1]:.2f} mm, Epsilon X {Xhz[2]:.3f}')
    print(f'Epsilon Y {Xhz[3]:.2f}, Gamma XY {Xhz[4]:.2f}, obrot wokol osi Z {Xhz[5]:.3f} mm/m')

