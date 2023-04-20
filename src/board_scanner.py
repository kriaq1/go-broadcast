from board import Board
from emulator import board_generator


class BoardScanner:
    async def get_board(self) -> (Board | None):
        return board_generator(100)
