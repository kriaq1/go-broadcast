import numpy as np
from collections import deque


class Move:
    def __init__(self, timestamp, x, y, color):
        self.timestamp: float = timestamp
        self.x: int = x
        self.y: int = y
        self.color: int = color


class ScenariosHandler:
    def __init__(self, confident_time=30 * 1000, ):
        self.moves_buffer: deque[Move] = deque()
        self.confident_moves: deque[Move] = deque()
        self.confident_time = confident_time
        self.buffer_size = 3


    def get_move(self):
        pass

    def handle_scenario(self, timestamp):
        if len(self.moves_buffer) == 0:
            return
        if timestamp - self.moves_buffer[0].timestamp >= self.confident_time:
            self.confident_moves.append(self.moves_buffer.popleft())
        if len(self.moves_buffer) >= self.buffer_size:
            pass
