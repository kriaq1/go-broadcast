import numpy as np
from .move import Move


def set_alive_recursively(board, x, y, is_dead):
    color = board[x][y]
    is_dead[x][y] = 0
    for delta in [-1, 1]:
        if 0 <= x + delta < board.shape[0] and board[x + delta][y] == color and is_dead[x + delta][y]:
            set_alive_recursively(board, x + delta, y, is_dead)
        if 0 <= y + delta < board.shape[1] and board[x][y + delta] == color and is_dead[x][y + delta]:
            set_alive_recursively(board, x, y + delta, is_dead)


def has_breath(board, x, y):
    for delta in [-1, 1]:
        if board.shape[0] > x + delta >= 0 == board[x + delta][y]:
            return True
        if board.shape[1] > y + delta >= 0 == board[x][y + delta]:
            return True
    return False


def find_captured(board: np.ndarray, color):
    result = []
    is_dead = np.ones(board.shape, dtype=bool)
    for x in range(board.shape[0]):
        for y in range(board.shape[1]):
            if board[x][y] == color and has_breath(board, x, y) and is_dead[x][y]:
                set_alive_recursively(board, x, y, is_dead)
    for x in range(board.shape[0]):
        for y in range(board.shape[1]):
            if board[x][y] == color and is_dead[x][y]:
                result.append((x, y))
    return result


def delete_captured(board, color):
    captured_list = find_captured(board, color)
    for x, y in captured_list:
        board[x][y] = 0
    return board


def check_correct(board, color=0):
    if color == 0:
        return check_correct(board, 1) and check_correct(board, -1)
    return len(find_captured(board, color)) == 0


def set_move(board, move: Move):
    if board[move.x][move.y] == 0:
        board[move.x][move.y] = move.color
        delete_captured(board, -move.color)
    return board
