from jednorodny.point import Point
from jednorodny.epoch import Epoch

# data = []


with open('data.txt', 'r') as f:
    e = Epoch('pomiar')
    for line in f:
        nr, cr, dx, dy, dz = line.split()
        point = Point(nr, cr, dx, dy, dz)
        e.add_point(point)

e.points_list()
