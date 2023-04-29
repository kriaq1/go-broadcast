import numpy as np
from collections import deque
from .types import Move
from .game_accumulator import GameAccumulator
from .board import Board


class ScenariosHandler:
    def __init__(self, game_accumulator: GameAccumulator,
                 board: Board = Board(),
                 current_player=-1,
                 confident_time=30 * 1000,
                 minimal_time_length=1 * 1000, ):
        # data
        self.board = board
        self.game_accumulator: GameAccumulator = game_accumulator
        self.moves_buffer: deque[Move] = deque()
        self.confident_moves: deque[Move] = deque()
        self.current_state = board.to_numpy()
        self.current_player = current_player
        # handle constants
        self.confident_time = confident_time
        self.minimal_time_length = minimal_time_length
        self.confident_buffer_size = 3

    def get_move(self) -> (Move | None):
        if len(self.confident_moves) == 0:
            return None
        else:
            return self.confident_moves.popleft()

    def validate(self, state: np.ndarray, prob: np.ndarray, timestamp: float):
        prob[state == 0 | state == self.current_state] = 0
        x, y = np.unravel_index(np.argmax(prob), prob.shape)
        self.current_state = state
        if not prob[x][y] == 0:
            self.moves_buffer.append(Move(timestamp=timestamp, x=x, y=y, color=state[x][y]))
        while self.handle_scenarios(timestamp):
            pass

    def handle_scenarios(self, timestamp: float) -> bool:  # returns bool to continue handle
        # buffer is empty
        if len(self.moves_buffer) == 0:
            return False
        # buffer has sufficient size
        if self.handle_confident_count(timestamp):
            return True
        # stone is standing too long
        return self.handle_exceeding_time_limit(timestamp)

    def handle_confident_count(self, timestamp: float) -> bool:
        if len(self.moves_buffer) < self.confident_buffer_size:
            return False
        # swap stones if 0th and 1st have same color
        if not self.game_accumulator.check_move(self.moves_buffer[0], self.moves_buffer[1].timestamp):
            return False
        self.swap_moves()

    def swap_moves(self):
        pass

    def handle_exceeding_time_limit(self, timestamp: float) -> bool:
        if timestamp - self.moves_buffer[0].timestamp < self.confident_time:
            return False

        until_time = timestamp
        # buffer has more 1 element we want to check until next move
        if len(self.moves_buffer) > 1:
            until_time = self.moves_buffer[1].timestamp
        if self.game_accumulator.check_move(self.moves_buffer[0], until_time):
            self.confident_moves.append(self.moves_buffer.popleft())
        return True
