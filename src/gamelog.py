from board import Board, Turn
import numpy as np


class GameLog:
    def __init__(self):
        self.board = Board()
        self.turns: list[Turn] = []
        self.turn_iterator = 0

    def apply_matrix(self, m: np.ndarray) -> bool:
        '''
        apply_matrix(m: np.ndarray) -> bool

        Tries to apply go matrix
        from recognition component in order
        to extract turn.

        Returns true, if extraction was complete,
        otherwise false
        '''
        assert m.ndim() == 2

        if self.board.to_numpy() == m:
            return False
        next_board = Board(array=m,
                           player=self.board.current_player.opposite())
        turn = self.board.get_turn(next_board)
        if turn:
            self.turns.append(turn)
            self.board = next_board
            return True
        return False

    def get_turn(self) -> Turn:
        '''
        get_turn() -> Turn

        Returns last unprocessed turn
        '''
        old_it = self.turn_iterator
        self.turn_iterator += 1
        return self.turns[old_it]
