import cv2
import time
from board import Board


def image_generator(filename):
    start_time = time.time()
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        return
    while True:
        current_time = time.time()
        cap.set(cv2.CAP_PROP_POS_MSEC, (current_time - start_time) * 1000)
        res, frame = cap.read()
        if res:
            yield frame
        else:
            return


class BoardGenerator:
    __slots__ = ["current_board", "countdown"]

    def __init__(self, countdown=20):
        self.current_board = Board()
        self.countdown = countdown

    def __call__(self, size, sleep_time=0) -> (Board | None):
        return Board()
        if self.countdown <= 0:
            return None
        self.current_board.put_stone(0, 0, 1)
        time.sleep(sleep_time)
        --self.countdown
        yield self.current_board

        self.current_board.put_stone(1, 1, -1)
        time.sleep(sleep_time)
        --self.countdown
        yield self.current_board
        self.current_board.clear()
        time.sleep(sleep_time)
        --self.countdown
        yield self.current_board


board_generator = BoardGenerator()
