class Move:
    def __init__(self, x, y, color, timestamp):
        self.x: int = x
        self.y: int = y
        self.color: int = color
        self.timestamp: float = timestamp

    def to_tuple(self) -> tuple[int, int, int, float]:
        return self.x, self.y, self.color, self.timestamp
