from deformations.point import Point


class Epoch:
    def __init__(self, name):
        self.name = name
        self.points = []

    def add_point(self, point: Point):
        self.points.append(point)

    def remove_point(self, point: Point):
        if point in self.points:
            self.points.remove(point)

    def points_list(self):
        for point in self.points:
            print(point)

    def __str__(self):
        return self.name

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Epoch):
            return False
        return self.name == o.name
