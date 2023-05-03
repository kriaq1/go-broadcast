from .game_validation import Move, GameValidation
import numpy as np
import asyncio


class GameLog:
    def __init__(self):
        self.game_validation = GameValidation()
        self.lock = asyncio.Lock()
        self.moves: list[Move] = []

    async def append_state(self, state: np.ndarray, prob: np.ndarray, quality: float, timestamp: float) -> bool:
        '''
        append_state
        '''
        try:
            async with self.lock:
                self.game_validation.validate(state, prob, quality, timestamp)
        except:
            return False
        return True

    async def get_move(self) -> (Move | None):
        '''
        get_move() -> Move

        Returns last unprocessed move
        '''
        while True:
            try:
                async with self.lock:
                    move = self.game_validation.get_move()
            except:
                return None
            if move:
                self.moves.append(move)
                return move
            else:
                await asyncio.sleep(0)
