import pytest
from src.board import Board
import numpy as np


@pytest.mark.test_board
def test_put_stone():
    board = Board(5)

    coordinates_black = [(0, 0), (0, 1), (1, 0), (2, 1), (2, 0), (0, 2), (1, 2), (2, 2)]
    for x, y in coordinates_black:
        board.put_stone(x, y, -1)

    coordinates_white = [(3, 0), (3, 1), (3, 2), (3, 3), (2, 3), (1, 3), (0, 3)]
    for x, y in coordinates_white:
        board.put_stone(x, y, 1)

    board.put_stone(1, 1, 1)

    coordinates_black = [(0, 0), (0, 1), (1, 0), (2, 1), (2, 0), (0, 2), (1, 2), (2, 2)]
    for x, y in coordinates_black:
        board.put_stone(x, y, -1)

    coordinates_black = [(x, 4) for x in range(5)] + [(4, y) for y in range(4)]
    for x, y in coordinates_black:
        board.put_stone(x, y, -1)

    true_board = np.array([[-1, -1, -1, 0, -1],
                           [-1, 0, -1, 0, -1],
                           [-1, -1, -1, 0, -1],
                           [0, 0, 0, 0, -1],
                           [-1, -1, -1, -1, -1]])

    assert (board.to_numpy() == true_board).all()


@pytest.mark.test_board
def test_check_correct():
    board = Board(19)
    board[0][1] = -1
    board[1][0] = -1
    assert board.check_correct(0)
    board[0][0] = 1
    assert board.check_correct(-1)
    assert not board.check_correct(1)


@pytest.mark.test_board
def test_find_captured():
    board = Board(19)
    board[0][1] = -1
    board[1][0] = -1
    assert not board.find_captured(1)
    assert not board.find_captured(-1)
    board[1][1] = 1
    assert not board.find_captured(1)
    board[0][0] = 1
    assert board.find_captured(1) == [(0, 0)]
    board[2][0] = 1
    board[0][2] = 1
    assert board.find_captured(1) == [(0, 0)]
    assert board.find_captured(-1) == [(0, 1), (1, 0)] or board.find_captured(-1) == [(1, 0), (0, 1)]


@pytest.mark.test_board
def test_clear():
    board = Board(19)
    board.put_stone(1, 1, 'b')
    assert board[1][1] == -1
    board.clear()
    assert board.size() == 19
    assert board[1][1] == 0
