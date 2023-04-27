import sente
from tkinter.filedialog import asksaveasfile


class SGF:
    def __init__(self):
        self.game = sente.Game(rules=sente.rules.JAPANESE)
        self.prev_color = 1

    def get_color(self, color):
        if color == 1:
            return sente.stone.WHITE
        else:
            return sente.stone.BLACK

    def play(self, x, y, color):
        if self.prev_color == color:
            self.game.pss()
        self.game.play(x, y, self.get_color(color))
        self.prev_color = color

    def save(self, path=None):
        if path is None:
            path = asksaveasfile().name
        sente.sgf.dump(self.game, path)

    def clear(self):
        self.game = sente.Game(rules=sente.rules.JAPANESE)
        self.prev_color = 1
