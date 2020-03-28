class Point:
    def __init__(self, number, x, y):
        self.number = number
        self.x = x
        self.y = y

    def __str__(self):
        return f'{self.number} {self.x} {self.y}'

    def __eq__(self, o: object) -> bool:
        if not isinstance(o, Point):
            return False
        return self.number == o.number and self.x == o.x and self.y == o.y