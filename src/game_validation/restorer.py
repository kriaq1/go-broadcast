from .container import BoardStateContainer
from .move import Move
import numpy as np
from .utils import check_correct, delete_captured


class MoveRestorer:
    def __init__(self, container: BoardStateContainer):
        self.container = container
        self.color = -1
        self.appearance_count = 3

    def preprocess_sequence(self, sequence, state_start, state_end):
        return sequence

    def restore(self, timestamp_start, timestamp_end, state_start, state_end) -> tuple[list[Move], list]:
        sequence = self.container.get(timestamp_start, timestamp_end)
        changes = np.where((state_end != state_start) & state_end != 0)
        changes = list(zip(*changes))
        changes_sequence = []
        for x, y in changes:
            index = self.first_appearance(x, y, state_end[x][y], sequence, 0, state_end)
            changes_sequence.append((x, y, index))
        changes_sequence = sorted(changes_sequence, key=lambda change: change[2])
        moves = []
        state = state_start.copy()
        for x, y, index in changes_sequence:
            try_state = state.copy()
            color = state_end[x][y]
            neighbours = self.get_neighbours(x, y)
            neighbours = [(x, y, self.first_appearance(x, y, -color, sequence, 0, state_end)) for (x, y) in neighbours]
            neighbours = list(filter(lambda n: n[2] is not None, neighbours))
            neighbours = sorted(neighbours, key=lambda n: n[2])
            for nx, ny, nindex in neighbours:
                prev_color = state[nx][ny]
                try_state[nx][ny] = -color
                if not check_correct(try_state):
                    try_state[nx][ny] = prev_color
                    continue
            try_state[x][y] = color
            delete_captured(try_state, -color)
            for nx, ny, nindex in neighbours:
                if try_state[nx][ny] == 0 and state_start[nx][ny] == 0:
                    moves.append(Move(x=nx, y=ny, color=-color, timestamp=sequence[nindex][2]))
                    state[nx][ny] = -color
            moves.append(Move(x=x, y=y, color=color,
                              timestamp=sequence[index][2] if index < len(sequence) else timestamp_end))
            state[x][y] = color
            delete_captured(state, -color)
        if moves:
            prev_color = self.color
            for i in range(len(moves) - 1):
                if moves[i].timestamp == moves[i + 1].timestamp and moves[i].color == prev_color:
                    moves[i], moves[i + 1] = moves[i + 1], moves[i]
                prev_color = moves[i].color
            self.color = moves[-1].color
        not_deleted = np.where(state != state_end)
        not_deleted = list(zip(*not_deleted))
        not_deleted = [(x, y, state[x][y]) for (x, y) in not_deleted]
        return moves, not_deleted

    def get_neighbours(self, x, y):
        shifts = [-1, 1, 0, 0]
        neighbors = []
        for shiftx, shifty in zip(shifts, reversed(shifts)):
            x1, y1 = x + shiftx, y + shifty
            if 0 <= x1 < 19 and 0 <= y1 < 19:
                neighbors.append((x1, y1))
        return neighbors

    def first_appearance(self, x, y, color, sequence, from_index, state_end) -> int:
        cnt = 0
        while from_index < len(sequence):
            if color == sequence[from_index][0][x][y]:
                cnt = cnt + 1
            else:
                cnt = 0
            from_index = from_index + 1
            if cnt >= self.appearance_count:
                return from_index
        if state_end[x][y] == color:
            return len(sequence)
