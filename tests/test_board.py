import pytest
from src.board import Board


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

    board.print_to_console()
