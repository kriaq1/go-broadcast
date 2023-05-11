import numpy as np
from .move import Move
from .container import BoardStateContainer
from .validator import BoardStateValidator
from .restorer import MoveRestorer


class GameValidation:
    def __init__(self, initial_state: np.ndarray):
        self.container = BoardStateContainer()
        self.validator = BoardStateValidator(self.container, initial_state=initial_state)
        self.restorer = MoveRestorer(self.container)
        self.move_groups = []
        self.index = 0
        self.group_index = 0
        self.delay = 1

        moves, not_deleted = self.restorer.restore(0, 0, np.zeros((19, 19), dtype=int), initial_state)
        if moves:
            self.move_groups.append(moves)

    def accept(self, state, prob, quality):
        if np.sum(prob == 0) > 50:
            return False
        return True

    def preprocess(self, state, prob):
        state = np.pad(state, ((1, 1), (1, 1)), 'constant', constant_values=0)
        prob = np.pad(prob, ((1, 1), (1, 1)), 'constant', constant_values=0)
        true_state = self.validator.get_true_state()[0]
        h, w = true_state.shape
        best = 0
        x, y = 0, 0
        for i in range(state.shape[0] - h + 1):
            for j in range(state.shape[1] - w + 1):
                accuracy = np.sum((state[i:i + h, j:j + w] == true_state) & (prob[i:i + h, j:j + w] != 0))
                if best < accuracy:
                    best = accuracy
                    x, y = i, j
        state = state[x: x + h, y: y + w]
        prob = prob[x: x + h, y: y + w]
        prob = expand_zeros(prob, 0.25)
        return state, prob

    def validate(self, state, prob, quality, timestamp):
        if not self.accept(state, prob, quality):
            return
        state, prob = self.preprocess(state, prob)
        self.container.push(state, prob, timestamp)
        prev_true_state, prev_timestamp = self.validator.get_true_state()
        true_state, timestamp = self.validator.update_true_state()
        if true_state is None:
            return
        moves, not_deleted = self.restorer.restore(prev_timestamp, timestamp, prev_true_state, true_state)
        if moves is None:
            pass
        if not_deleted:
            ok = np.zeros(len(not_deleted), dtype=bool)
            remaining = []
            for i in range(1, min(self.delay, len(self.move_groups)) + 1):
                remaining.append([])
                for move in self.move_groups[-i]:
                    keep = True
                    for index, stone in enumerate(not_deleted):
                        if not ok[index] and stone[0] == move.x and stone[1] == move.y and stone[2] == move.color:
                            ok[index] = True
                            keep = False
                            break
                    if keep:
                        remaining[-1].append(move)
            if np.all(ok):
                for i in range(1, min(self.delay, len(self.move_groups)) + 1):
                    self.move_groups[-i] = remaining[i - 1]
                for i in reversed(list(range(1, min(self.delay, len(self.move_groups)) + 1))):
                    if not self.move_groups[-i]:
                        del self.move_groups[-i]
            else:
                moves = []
        if moves:
            self.move_groups.append(moves)

    def get_move(self):
        if self.group_index + self.delay >= len(self.move_groups):
            return None
        move = self.move_groups[self.group_index][self.index]
        self.index = self.index + 1
        if self.index == len(self.move_groups[self.group_index]):
            self.index = 0
            self.group_index = self.group_index + 1
        return Move(x=move.y + 1, y=move.x + 1, color=move.color, timestamp=move.timestamp)

    def get_last_move_groups(self):
        if self.move_groups and self.delay:
            result = self.move_groups[-min(self.delay, len(self.move_groups)):]
            return [Move(x=move.y + 1, y=move.x + 1, color=move.color, timestamp=move.timestamp) for moves in result for
                    move in moves]
        return []

    def update_parameters(self,
                          delay: int = 1,
                          appearance_count: int = 3,
                          valid_time: int = 3 * 1000,
                          valid_zeros_count: int = 20,
                          valid_thresh: float = 0.2, ):
        self.delay = delay
        self.restorer.appearance_count = appearance_count
        self.validator.valid_time = valid_time
        self.validator.valid_zeros_count = valid_zeros_count
        self.validator.valid_thresh = valid_thresh


def expand_zeros(prob, coef):
    mask = np.pad((prob == 0), ((1, 1), (1, 1)), 'constant', constant_values=False)
    result = mask.copy()
    for shift0 in [-1, 0, 1]:
        for shift1 in [-1, 0, 1]:
            rolled_mask = np.roll(np.roll(mask, shift=shift0, axis=0), shift=shift1, axis=1)
            result = result | rolled_mask
    prob[result[1:-1, 1:-1]] *= coef
    return prob
