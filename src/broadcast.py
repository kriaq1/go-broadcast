from board import Board
from api import API
from board_scanner import BoardScanner


class Broadcast:
    __slots__ = ["api", "app", "scanner"]

    def __init__(self, api: [API], *scanner_args):
        self.scanner = BoardScanner(*scanner_args)
        self.api = api

    def start(self):
        b = self.scanner.get_board()
        while b:
            for api in self.api:
                try:
                    api.add(b)
                    api.broadcast()
                except Exception:
                    pass
            b = self.scanner.get_board()


def main():
    from asciiapi import ASCIIDump
    broadcaster = Broadcast([ASCIIDump("test")])
    broadcaster.start()


main()
