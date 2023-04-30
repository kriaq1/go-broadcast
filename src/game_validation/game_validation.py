import numpy as np
from .game_accumulator import GameAccumulator
from .scenarios_handler import ScenariosHandler


class GameValidation:
    def __init__(self, board: np.ndarray = np.zeros((19, 19), dtype=int),
                 current_player=-1,
                 accumulating_time=2 * 1000,
                 accumulating_thresh=0.4,
                 max_empties_count=50,
                 max_border_empties_count=12,
                 preprocess_coef=0.5,
                 check_move_thresh=0.0,
                 confident_time=30 * 1000,
                 minimal_time_length=1 * 1000,
                 ):
        self.game_accumulator = GameAccumulator(accumulating_time=accumulating_time,
                                                accumulating_thresh=accumulating_thresh,
                                                max_empties_count=max_empties_count,
                                                max_border_empties_count=max_border_empties_count,
                                                preprocess_coef=preprocess_coef,
                                                check_move_thresh=check_move_thresh)
        self.scenarios_handler = ScenariosHandler(game_accumulator=self.game_accumulator,
                                                  board=board,
                                                  current_player=current_player,
                                                  confident_time=confident_time,
                                                  minimal_time_length=minimal_time_length)

    def validate(self, state: np.ndarray, prob: np.ndarray, quality: float, timestamp: float):
        self.game_accumulator.accumulate(state=state, prob=prob, timestamp=timestamp)
        accumulated = self.game_accumulator.get_accumulated()
        if accumulated is None:
            return
        self.scenarios_handler.validate(*accumulated)

    def get_move(self) -> (tuple[int, int, int, float] | None):
        move = self.scenarios_handler.get_move()
        if move is not None:
            self.game_accumulator.pop_until(move.timestamp)
            return move.to_tuple()
        else:
            return None
