from collections import deque
from bisect import bisect_left, bisect_right

import numpy as np


class BoardStateContainer:
    def __init__(self):
        self.timestamp_deque: deque[float] = deque()
        self.deque: deque[tuple[np.ndarray, np.ndarray, float]] = deque()

    def push(self, state, prob, timestamp):
        assert not self.deque or self.deque[-1][2] <= timestamp
        self.timestamp_deque.append(timestamp)
        self.deque.append((state, prob, timestamp))

    def get(self, timestamp_start, timestamp_end=None):
        if timestamp_end is None:
            timestamp_end = self.timestamp_deque[-1]
        if not self.timestamp_deque:
            return []
        start = bisect_left(self.timestamp_deque, timestamp_start)
        end = bisect_right(self.timestamp_deque, timestamp_end)
        return list(self.deque)[start:end]

    def pop_left(self, timestamp):
        while self.timestamp_deque and self.timestamp_deque[0] < timestamp:
            self.timestamp_deque.popleft()
            self.deque.popleft()

    def last(self):
        return self.deque[-1]
