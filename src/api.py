from board import Turn


class API:
    def add(self, turn: Turn):
        raise NotImplementedError()

    def broadcast(self):
        raise NotImplementedError()
