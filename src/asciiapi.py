from api import API
from board import Board, Turn
import asyncio


class ASCIIDump(API):
    '''
    A basic API that represents dumping board
    state into file through ASCII pseudographics
    '''
    __slots__ = ["filepath", "board", "turns", "lock"]

    def __init__(self, path: str):
        self.filepath = path
        self.turns: list[Turn] = []
        self.board = Board()
        self.lock = asyncio.Lock()

    async def add(self, turn: Turn):
        self.turns.append(turn)

    async def broadcast(self):
        async with self.lock:
            f = open(self.filepath, "a")
            print("", file=f)
            for turn in self.turns:
                self.board.apply_turn(turn)
                self.board.print_to_console(f)
            self.turns.clear()
            f.close()
