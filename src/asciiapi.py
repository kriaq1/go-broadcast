from api import API
from board import Board
import asyncio


class ASCIIDump(API):
    '''
    A basic API that represents dumping board
    state into file through ASCII pseudographics
    '''
    __slots__ = ["filepath", "boards", "lock"]

    def __init__(self, path: str):
        self.filepath = path
        self.boards: list[Board] = []
        self.lock = asyncio.Lock()

    async def add(self, board: Board):
        self.boards.append(board)

    async def broadcast(self):
        async with self.lock:
            f = open(self.filepath, "a")
            # print("", file=f)
            for board in self.boards:
                board.print_to_console(f)
            f.close()
