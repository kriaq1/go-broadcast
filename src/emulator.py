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


def board_generator(size, sleep_time):
    pass
