class Point:
    def __init__(self, nr, perimeter, dx, dy, dz, x=0, y=0):
        self.nr = nr
        self.perimeter = perimeter
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.x = x
        self.y = y

    def __str__(self):
        return f'{self.nr} {self.dx} {self.dy} {self.dz} {self.x} {self.y}'

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Point):
            return False
        return self.nr == o.nr and self.perimeter == o.perimeter and self.dx == o.dx and self.dy == o.dy and self.dz == o.dz
