import numpy as np


class Move:
    def __init__(self, timestamp, x, y, color):
        self.timestamp: float = timestamp
        self.x: int = x
        self.y: int = y
        self.color: int = color
