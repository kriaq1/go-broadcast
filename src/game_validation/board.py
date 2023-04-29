import numpy as np
from .types import Move


class Board:
    def __init__(self, state: np.ndarray):
        self._board = state

    def clear(self, new_size=None):
        if new_size is None:
            new_size = self.size()
        self._board = np.zeros((new_size, new_size), dtype=int)

    def size(self):
        return self._board.shape[0]

    def check_move(self, move: Move) -> bool:
        if self[move.x][move.y] != 0:
            return False
        copy = self.copy()
        copy.set_move(move)
        return copy.check_correct()

    def check_moves(self, moves: list[Move]) -> int:
        copy = self.copy()
        for i, move in enumerate(moves):
            if self[move.x][move.y] != 0:
                return i
            copy.set_move(move)
            if not copy.check_correct():
                return i
        return -1

    def set_move(self, move: Move):
        self[move.x][move.y] = move.color
        self.delete_captured(-move.color)

    def set_alive_recursively(self, x, y, is_dead):
        color = self[x][y]
        is_dead[x][y] = 0
        for delta in [-1, 1]:
            if 0 <= x + delta < self.size() and self[x + delta][y] == color and is_dead[x + delta][y]:
                self.set_alive_recursively(x + delta, y, is_dead)
            if 0 <= y + delta < self.size() and self[x][y + delta] == color and is_dead[x][y + delta]:
                self.set_alive_recursively(x, y + delta, is_dead)

    def has_breath(self, x, y):
        for delta in [-1, 1]:
            if self.size() > x + delta >= 0 == self[x + delta][y]:
                return True
            if self.size() > y + delta >= 0 == self[x][y + delta]:
                return True
        return False

    def find_captured(self, color):
        result = []
        is_dead = np.ones((self.size(), self.size()), dtype=bool)
        for x in range(self.size()):
            for y in range(self.size()):
                if self[x][y] == color and self.has_breath(x, y) and is_dead[x][y]:
                    self.set_alive_recursively(x, y, is_dead)
        for x in range(self.size()):
            for y in range(self.size()):
                if self[x][y] == color and is_dead[x][y]:
                    result.append((x, y))
        return result

    def delete_captured(self, color):
        captured_list = self.find_captured(color)
        for x, y in captured_list:
            self[x][y] = 0

    def check_correct(self, color=0):
        if color == 0:
            return self.check_correct(1) and self.check_correct(-1)
        return len(self.find_captured(color)) == 0

    def __getitem__(self, key):
        return self._board[key]

    def __len__(self):
        return self.size()

    def copy(self):
        return Board(self._board.copy())

    def to_numpy(self):
        return self._board.copy()

    def to_list(self):
        return self._board.tolist()
