import numpy as np
from collections import deque
from .types import Move
import bisect


class GameAccumulator:
    """
    Hyperparameters:
        accumulated_time,
        accumulated_thresh,
        max_empties_count,
        max_border_empties_count,
        preprocess_coef,
        check_move_thresh
    Basic methods:
        accumulate, that receives states and saves it after validation;
        get_accumulated returns accumulated states and probabilities;
        check_move checks that stone was standing until some time
    """
    def __init__(self, accumulating_time=5 * 1000,
                 accumulating_thresh=0.4,
                 max_empties_count=50,
                 max_border_empties_count=12,
                 preprocess_coef=0.5,
                 check_move_thresh=0.0):
        # buffers
        self.state_buffer: deque[np.ndarray] = deque()
        self.prob_buffer: deque[np.ndarray] = deque()
        self.timestamps_buffer: deque[float] = deque()
        # current state for accumulating
        self.current: int = 0
        # get_accumulated_state constants
        self.accumulating_time = accumulating_time
        self.accumulating_thresh = accumulating_thresh
        # preprocess constants
        self.preprocess_coef = preprocess_coef
        # check_validity constants
        self.max_zeros_count = max_empties_count
        self.max_border_zeros_count = max_border_empties_count
        # check_move constants
        self.check_move_thresh = check_move_thresh

    def check_validity(self, state: np.ndarray, prob: np.ndarray, timestamp: float) -> bool:
        zeros_count = np.sum(prob == 0)
        border_zeros_count = np.sum(prob[0:] == 0) + np.sum(prob[-1:] == 0) + np.sum(prob[:0] == 0) + np.sum(
            prob[:-1] == 0)
        return zeros_count <= self.max_zeros_count and border_zeros_count <= self.max_border_zeros_count

    def preprocess(self, state: np.ndarray, prob: np.ndarray, timestamp: float) -> tuple[np.ndarray, np.ndarray, float]:
        mask = np.pad((prob == 0), ((1, 1), (1, 1)), 'constant', constant_values=0)
        for shift0 in [-1, 0, 1]:
            for shift1 in [-1, 0, 1]:
                if shift0 == 0 == shift1:
                    continue
            rolled_mask = np.roll(np.roll(mask, shift=shift0, axis=0), shift=shift1, axis=1)
            mask = mask | rolled_mask
        prob[mask[1:-1, 1:-1]] *= self.preprocess_coef
        return state, prob, timestamp

    def accumulate(self, state: np.ndarray, prob: np.ndarray, timestamp: float):
        if self.check_validity(state, prob, timestamp):
            state, prob, timestamp = self.preprocess(state, prob, timestamp)
            self.state_buffer.append(state)
            self.prob_buffer.append(to3dim_prob(state, prob))
            self.timestamps_buffer.append(timestamp)

    def state_prob_by_3dim(self, accumulated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state = np.argmax(accumulated, axis=0) - 1
        prob = np.max(accumulated, axis=0)
        return state, prob

    def accumulation(self, probs: np.ndarray) -> np.ndarray:
        return np.mean(probs, axis=0)

    def get_accumulated(self) -> (tuple[np.ndarray, np.ndarray, float] | None):
        cur_timestamp = self.timestamps_buffer[self.current]
        if len(self.state_buffer) == 0 or \
                self.timestamps_buffer[-1] < self.accumulating_time + cur_timestamp:
            return
        l, r = self.current, bisect.bisect_right(self.timestamps_buffer, cur_timestamp + self.accumulating_time)
        accumulated = self.accumulation(np.array(self.prob_buffer[l:r]))
        accumulated = np.where(accumulated >= self.accumulating_thresh, accumulated, 0)
        self.current += 1
        return self.state_prob_by_3dim(accumulated) + (cur_timestamp,)

    def check_move(self, move: Move, timestamp: float) -> bool:
        l = bisect.bisect_right(self.timestamps_buffer, move.timestamp)
        r = bisect.bisect_left(self.timestamps_buffer, timestamp)
        move_probs = np.array(prob[move.x][move.y] for prob in self.prob_buffer[l:r])
        accumulated = np.mean(move_probs, axis=0)
        return np.max(accumulated) > self.check_move_thresh and np.argmax(accumulated) - 1 == move.color

    def pop_until(self, timestamp: float):
        i = 0
        while len(self.timestamps_buffer) != 0 and self.timestamps_buffer[i] <= timestamp:
            self.state_buffer.popleft()
            self.timestamps_buffer.popleft()
            self.prob_buffer.popleft()
            i += 1
        self.current = max(self.current - i, 0)


def to3dim_prob(state: np.ndarray, prob: np.ndarray) -> np.ndarray:
    return np.array(np.where(state == c, prob, 0) for c in (-1, 0, 1))
