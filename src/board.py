import numpy as np
from enum import Enum
from sys import stdout


class GoBoardException(Exception):
    pass


class OccupiedException(GoBoardException):
    pass


class ForbiddenMoveException(GoBoardException):
    pass


class OutOfBoundsException(GoBoardException):
    pass


class Player(Enum):
    White = 1
    Black = -1

    def opposite(self):
        return Player.White if self == Player.Black else Player.Black


class Turn:
    def __init__(self, x: int, y: int, player: Player):
        self.x = x
        self.y = y
        self.player = player


# board[x][y] == -1 - black color, 1 - white color, 0 - empty
class Board:
    def __init__(self, size=19, array=None, player=Player.White):
        if array is None:
            self._board = np.zeros((size, size), dtype=int)
        else:
            self._board = np.array(array)
            shape = self._board.shape
            assert len(shape) == 2 and shape[0] == shape[1]
        self.current_player = player

    def clear(self, new_size=None):
        if new_size is None:
            new_size = self.size()
        self._board = np.zeros((new_size, new_size), dtype=int)

    def size(self):
        return self._board.shape[0]

    def is_in_bounds(self, x: int, y: int):
        return 0 <= x < self.size() and 0 <= y < self.size()

    # Ставит камень указанного цвета на доску, проверяя на корректность ход и удаляя захваченные камни
    # Не проверяет повторение предыдущей позиции
    def put_stone(self, x, y, color):
        if not self.is_in_bounds(x, y):
            raise OutOfBoundsException()
        if self[x][y] != 0:
            raise OccupiedException()

        color = self.get_color_value(color)
        self[x][y] = color
        self.delete_captured(-color)
        if not self.check_correct(color):
            raise ForbiddenMoveException()

    def get_color_value(self, color: (str | int)):
        if type(color) == str:
            color = color.lower()[0]
            assert color == 'b' or color == 'w'
            color = 1 if color == 'w' else -1
        assert color == 1 or color == -1
        return color

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
            if 0 <= x + delta < self.size() and self[x + delta][y] == 0:
                return True
            if 0 <= y + delta < self.size() and self[x][y + delta] == 0:
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

    def get(self, coordinate='S19'):
        x, y = self.get_coordinates(coordinate)
        return self._board[x][y]

    def get_coordinates(self, coordinate):
        x = ord(coordinate[0].lower()) - ord('a')
        y = coordinate[1:]
        y = int(y) if y.isdigit() else ord(y.lower()) - ord('a')
        assert 0 <= x < self.size()
        assert 0 <= y < self.size()
        return x, y

    def print_to_console(self, output=stdout):
        board = self.to_numpy().astype(str)
        for x in range(self.size()):
            for y in range(self.size()):
                if board[x][y] == '1':
                    board[x][y] = 'w'
                elif board[x][y] == '-1':
                    board[x][y] = 'b'
                else:
                    board[x][y] = '.'
            print(
                chr(ord('a') + x),
                np.array2string(board[x], formatter={'str_kind': lambda x: x}),
                file=output
            )
        print(' ', np.arange(1, self.size() + 1), file=output)

    def make_turn(self, coordinate: (tuple[int, int] | str)):
        color = self.current_player.value
        if isinstance(coordinate, str):
            coordinate = self.get_coordinates(coordinate)
        x, y = coordinate
        self.put_stone(x, y, color)
        self.current_player = self.current_player.opposite()

    def get_turn(self, next_state) -> (Turn | None):
        for x in range(self.size()):
            for y in range(self.size()):
                nxt = next_state._board[x][y]
                cur = self._board[x][y]
                if cur == 0 and nxt == self.current_player.value:
                    return Turn(x, y, self.current_player)
        return None

    def __getitem__(self, key):
        return self._board[key]

    def __len__(self):
        return self.size()

    def copy(self):
        return Board(self.size(), self._board)

    def to_numpy(self):
        return self._board.copy()

    def to_list(self):
        return self._board.tolist()

