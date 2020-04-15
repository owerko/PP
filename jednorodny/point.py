class Point:
    def __init__(self, nr, circuit, dx, dy, dz):
        self.nr = nr
        self.circuit = circuit
        self.dx = dx
        self.dy = dy
        self.dz = dz

    def __str__(self):
        return f'{self.nr} {self.circuit} {self.dx} {self.dy} {self.dz}'

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Point):
            return False
        return self.nr == o.nr and self.circuit == o.circuit and self.dx == o.dx and self.dy == o.dy and self.dz == o.dz
