import numpy as np
from .utils import check_correct
from .container import BoardStateContainer


class BoardStateValidator:
    def __init__(self, container: BoardStateContainer, initial_state: np.ndarray):
        self.container = container
        self.poor_timestamp = 0
        self.timestamp = 0
        self.true_state = initial_state.copy()
        self.valid_time = 3 * 1000
        self.valid_zeros_count = 20
        self.valid_thresh = 0.2

    def state_probs_to_3dim(self, state, prob):
        return np.array([prob * (state == -1), prob * (state == 0), prob * (state == 1)])

    def find_true_state(self, sequence):
        if not sequence or self.poor_timestamp + self.valid_time > sequence[-1][2]:
            return None, 0
        index = 0
        while sequence[index][2] <= self.poor_timestamp:
            index += 1
        if index + 1 < len(sequence):
            self.poor_timestamp = sequence[index + 1][2]
        probs = [self.state_probs_to_3dim(state, prob) for state, prob, timestamp in sequence[index:]]
        mean = np.mean(probs, axis=0)
        mean = np.where(mean >= self.valid_thresh, mean, 0)
        if np.sum(np.max(mean, axis=0) == 0) > self.valid_zeros_count:
            return None, 0
        meaning_state = np.argmax(mean, axis=0) - 1
        meaning_state[np.all(mean == 0, axis=0)] = 0
        state, prob, timestamp = sequence[index]
        if np.sum(prob == 0) > self.valid_zeros_count:
            return None, 0
        state[prob == 0] = self.true_state[prob == 0]

        if np.all(meaning_state == state) and check_correct(state) and np.any(state != self.true_state):
            return meaning_state, sequence[index][2]

        return None, 0

    def update_true_state(self) -> tuple[np.ndarray | None, float]:
        sequence = self.container.get(self.timestamp, None)
        true_state, timestamp = self.find_true_state(sequence)
        if true_state is None:
            return None, 0
        self.true_state = true_state
        self.timestamp = timestamp
        return self.true_state, self.timestamp

    def get_true_state(self):
        return self.true_state, self.timestamp

    def change_true_state(self, state, timestamp):
        self.true_state = state
        self.timestamp = timestamp
