from api import API
from board import Board


class ASCIIDump(API):
    '''
    A basic API that represents dumping board
    state into file through ASCII pseudographics
    '''
    __slots__ = [filepath, boards]

    def __init__(self, path: str):
        self.filepath = path
        self.boards = []

    def add(self, board: Board):
        self.boards.append(board)

    def broadcast(self):
        f = open(self.path, "a")
        print("", file=f)
        for board in self.boards:
            board.print_to_console(f)
        f.close()
