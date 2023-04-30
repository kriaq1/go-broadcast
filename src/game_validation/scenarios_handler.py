import numpy as np
from collections import deque
from .types import Move
from .game_accumulator import GameAccumulator
from .board import Board
from enum import Enum


class Scenario(Enum):
    STOP_HANDLE = 1
    SKIP_MOVE = 2
    SWAP_MOVES_AND_ADD = 3
    SKIP_NEXT_MOVE_AND_ADD = 4
    ADD = 5


class ScenariosHandler:

    def __init__(self, game_accumulator: GameAccumulator,
                 board: np.ndarray,
                 current_player=-1,
                 confident_time=30 * 1000,
                 minimal_time_length=1 * 1000, ):
        # data
        self.board = Board(board)
        self.game_accumulator: GameAccumulator = game_accumulator
        self.moves_buffer: deque[Move] = deque()
        self.confident_moves: deque[Move] = deque()
        self.last_state = board
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
        prob[(state == 0) | (state == self.last_state)] = 0
        x, y = np.unravel_index(np.argmax(prob), prob.shape)
        self.last_state = state
        if not prob[x][y] == 0:
            self.moves_buffer.append(Move(timestamp=timestamp, x=x, y=y, color=state[x][y]))
        while self.handle_scenarios(timestamp):
            pass

    def handle_scenarios(self, timestamp: float) -> bool:  # returns True to continue handle
        scenario = self.get_scenario(timestamp)
        if scenario == Scenario.STOP_HANDLE:
            return False
        if scenario == scenario.SKIP_MOVE:
            self.moves_buffer.popleft()
        if scenario == scenario.SWAP_MOVES_AND_ADD:
            self.moves_buffer[0], self.moves_buffer[1] = self.moves_buffer[1], self.moves_buffer[0]
            self.append_confident_move(self.moves_buffer.popleft())
        if scenario == scenario.ADD:
            self.append_confident_move(self.moves_buffer.popleft())
        return True

    def get_scenario(self, timestamp) -> Scenario:
        if len(self.moves_buffer) == 0:
            return Scenario.STOP_HANDLE
        if len(self.moves_buffer) < self.confident_buffer_size and \
                timestamp - self.moves_buffer[0].timestamp < self.confident_time:
            return Scenario.STOP_HANDLE

        until_time = timestamp
        # buffer has more 1 element we want to check until next move
        if len(self.moves_buffer) > 1:
            until_time = self.moves_buffer[1].timestamp
        if not self.game_accumulator.check_move(self.moves_buffer[0], until_time):
            return Scenario.SKIP_MOVE
        if len(self.moves_buffer) < self.confident_buffer_size:
            return Scenario.ADD

        if self.moves_buffer[1].color == self.current_player == -self.moves_buffer[0].color and \
                self.game_accumulator.check_move(self.moves_buffer[1], self.moves_buffer[2].timestamp) and \
                self.board.check_moves([self.moves_buffer[1], self.moves_buffer[0]]) == -1:
            return Scenario.SWAP_MOVES_AND_ADD
        if self.board.check_move(self.moves_buffer[0]):
            return Scenario.ADD
        else:
            return Scenario.SKIP_MOVE

    def append_confident_move(self, move: Move):
        self.board.set_move(move)
        self.current_player = -move.color
        self.confident_moves.append(move)
