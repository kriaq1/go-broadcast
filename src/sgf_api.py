import sente
from tkinter.filedialog import asksaveasfile


class SGFWriter:
    def __init__(self):
        self.game = sente.Game(rules=sente.rules.JAPANESE)
        self.prev_color = 1

    def add(self, x, y, color):
        if self.prev_color == color:
            self.game.pss()
        self.game.play(x, y, self.get_color(color))
        self.prev_color = color

    def pss(self):
        self.game.pss()
        self.prev_color = -self.prev_color

    def save(self, path=None):
        if path is None:
            file = asksaveasfile()
            path = file.name if file else None
        else:
            with open(path, 'w') as _:
                pass
        if path is not None:
            sente.sgf.dump(self.game, path)

    def clear(self):
        self.game = sente.Game(rules=sente.rules.JAPANESE)
        self.prev_color = 1

    def is_legal(self, x, y, color):
        return self.game.is_legal(x, y, self.get_color(color))

    def change_move(self, index, x=None, y=None):
        if x is None or y is None:
            x = 20
            y = 20
        sequence = self.game.get_sequence()
        sequence[index] = sente.Move(x - 1, y - 1, sequence[index].get_stone())
        try:
            game = sente.Game(rules=sente.rules.JAPANESE)
            game.play_sequence(sequence)
            self.game = game
            return True
        except Exception:
            return False

    def get_sequence(self):
        sequence = []
        for move in self.game.get_sequence():
            if move.get_x() == 19:
                continue
            color = -1 if move.get_stone() == sente.stone.BLACK else 1
            sequence.append((move.get_x() + 1, move.get_y() + 1, color))
        return sequence

    def get_color(self, color):
        if color == 1:
            return sente.stone.WHITE
        else:
            return sente.stone.BLACK


class SGFAPI:
    def __init__(self, save_path: str = None):
        self.sgf = SGFWriter()
        self.save_path = save_path

    def add(self, move):
        self.sgf.add(move.x, move.y, move.color)

    def broadcast(self):
        self.sgf.save(self.save_path)
