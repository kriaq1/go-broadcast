import numpy as np
from collections import deque
from .board import Board


class SimpleGameValidation:
    def __init__(self, board=Board()):
        self.board = board
        self.board_size = 19
        self.valid_border_zeros_cnt = 15  # bound of zeros count on border
        self.valid_zeros_cnt = 30
        self.valid_segm_quality = 0  # quality bound
        self.min_buff_size = 0  # necessary states for moves recognition
        self.time_limit = 5 * 1000  # time(ms) distance between recognized moves and current state
        self.probability_thresh = 0.5
        self.states_buff: deque[
            tuple[np.ndarray, np.ndarray, float]] = deque()  # save tuples of board, prob and timestamp
        self.prob_accumulator = np.zeros((3, self.board_size, self.board_size),
                                         dtype=float)  # save sum of probs in buff for 3 classes
        self.moves_buff: deque[list[tuple[int, int, int, float]]] = deque()  # save moves

    def validate(self, board: np.ndarray, prob: np.ndarray, quality: float, timestamp: float):
        if not self.check_validity(prob, quality):
            return
        board, prob = self.preprocess_state(board, prob)
        self.states_buff.append((board, prob, timestamp))
        self.accumulate(board, prob, 1)
        while self.check_validation_quality():
            self.recognize_changes(timestamp)
            board, prob, timestamp = self.states_buff.popleft()
            self.accumulate(board, prob, -1)

    def get_move(self) -> (tuple[int, int, int] | None):
        if len(self.moves_buff) == 0:
            return None
        if len(self.moves_buff[0]) == 1:
            return self.moves_buff.popleft()[0][:3]

    def check_validity(self, prob: np.ndarray, quality: float):
        border_zeros_cnt = np.sum(prob[0, :] == 0) + np.sum(prob[self.board_size - 1, :] == 0) + np.sum(
            prob[:, 0] == 0) + np.sum(prob[:, self.board_size - 1] == 0)
        zeros_cnt = np.sum(prob == 0)
        return border_zeros_cnt <= self.valid_border_zeros_cnt and zeros_cnt <= self.valid_zeros_cnt \
            and self.valid_segm_quality <= quality

    def preprocess_state(self, board: np.ndarray, prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return board, prob

    def check_validation_quality(self):
        states_cnt = len(self.states_buff)
        return states_cnt >= self.min_buff_size and \
            (self.states_buff[states_cnt - 1][2] - self.states_buff[0][2]) > self.time_limit

    def recognize_changes(self, timestamp: float):
        state = self.board.to_numpy()
        next_state = self.states_buff[0][0]
        max_probs = np.max(np.array(tuple(p for _, p, __ in self.states_buff)), axis=0)
        max_probs[max_probs == 0] = 1
        prob = self.prob_accumulator / max_probs
        prob_state = np.argmax(prob, axis=0) - 1
        prob = np.max(prob, axis=0)
        mask = (prob_state == next_state) & (state == 0) & (next_state != 0) & (
                prob > self.probability_thresh * len(self.states_buff))
        prob[~mask] = -1
        for index in np.argsort(prob.reshape(-1))[::-1][:np.sum(mask)]:
            x, y = np.unravel_index(index, prob.shape)
            color = next_state[x][y]
            self.moves_buff.append([(x, y, color, timestamp)])
            self.board.put_stone(x, y, color)
            break

    def accumulate(self, board, prob, coefficient):
        self.prob_accumulator += coefficient * np.vstack([board == -1, board == 0, board == 1]).reshape(
            self.prob_accumulator.shape) * prob
