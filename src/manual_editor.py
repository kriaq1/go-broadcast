from board import Board


class ManualEditor:
    def __init__(self):
        self.board = Board()

    def add(self, board):
        self.board = board

    def put_stone(self, x, y, color):
        self.board._board[x][y] = color

    def remove_stone(self, x, y):
        self.board._board[x][y] = 0

    def get(self):
        return self.board

    def send(self, api):
        return api.add(self.board)
