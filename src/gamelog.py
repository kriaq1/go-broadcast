from .game_validation import Move, GameValidation
from .game_validation import utils
import numpy as np
import asyncio


class GameLog:
    def __init__(self, initial_state=np.zeros((19, 19), dtype=int)):
        self.game_validation = GameValidation(initial_state)
        self.moves: list[Move] = []
        self.state = initial_state.copy()
        self.val = np.zeros((19, 19), dtype=int)

    def append_state(self, state: np.ndarray, prob: np.ndarray, quality: float, timestamp: float):
        '''
        append_state
        '''
        try:
            self.game_validation.validate(state, prob, quality, timestamp)
        except Exception:
            pass

    async def get_move(self) -> (Move | None):
        '''
        get_move() -> Move

        Returns last unprocessed move
        '''
        while True:
            try:
                move = self.game_validation.get_move()
            except Exception:
                return None
            if move is not None:
                self.moves.append(move)
                self.state = utils.set_move(self.state, move)
                self.val[move.y - 1][move.x - 1] = len(self.moves)
                return move
            await asyncio.sleep(0)

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        return self.state, self.val

    def update_parameters(self,
                          initial_state: np.ndarray = None,
                          **kwargs):
        if initial_state is not None:
            self.game_validation = GameValidation(initial_state)
            self.moves = []
            self.state = initial_state.copy()
            self.val = np.zeros((19, 19), dtype=int)
        self.game_validation.update_parameters(**kwargs)
