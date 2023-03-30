from board import Board
from emulator import board_generator


class BoardScanner:
    def get_board(self) -> Board:
        return board_generator(100)
