from board import Board
from api import API
from board_scanner import BoardScanner

app = Flask(__name__)


class Broadcast:
    __slots__ = [api, app, scanner]

    def __init__(self, api: [API], *scanner_args):
        self.app = Flask(__name__)
        self.scanner = BoardScanner(*scanner_args)

    def start():
        b = self.scanner.get_board()
        while b:
            for api in self.api:
                try:
                    api.add(b)
                    api.broadcast()
                except:
                    pass
            b = self.scanner.get_board()


def main():
    from asciiapi import ASCIIDump
    broadcaster = Broadcast([ASCIIDump("test")])
    broadcaster.start()


main()
