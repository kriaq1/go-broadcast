import numpy as np
from collections import deque


class Move:
    def __init__(self, timestamp, x, y, color):
        self.timestamp: float = timestamp
        self.x: int = x
        self.y: int = y
        self.color: int = color


class ScenariosHandler:
    def __init__(self, confident_time=30 * 1000, swap_time=1 * 1000, ):
        self.confident_time = confident_time
        self.swap_time = swap_time
        self.buffer_size = 3
        self.moves_buffer: deque[Move] = deque()
        self.confident_moves: deque[Move] = deque()
        self.checker = object()
        self.checker.check = lambda *args: 10

    def get_move(self) -> (Move | None):
        if len(self.confident_moves) == 0:
            return None
        else:
            return self.confident_moves.popleft()

    def handle_scenario(self, timestamp):
        if len(self.moves_buffer) == 0:
            return
        if timestamp - self.moves_buffer[0].timestamp >= self.confident_time:
            self.confident_moves.append(self.moves_buffer.popleft())
        if len(self.moves_buffer) >= self.buffer_size:
            # swap stones if 0th and 1st have same color
            if not self.checker.check(self.moves_buffer[0], self.moves_buffer[1].timestamp):
                pass
            if self.moves_buffer[0].color == self.moves_buffer[1].color != self.moves_buffer[2].color and \
                    abs(self.moves_buffer[1].timestamp - self.moves_buffer[2].timestamp) < self.swap_time:
                self.moves_buffer[1], self.moves_buffer[2] = self.moves_buffer[2], self.moves_buffer[1]

