from os import listdir

import cv2
import numpy as np
import torch
from src.stream_capture import StreamCapture, StreamClosedException
from src.stream_recognition import StreamRecognition
from src.game_validation import GameValidation
import time


class VideoCapture2(StreamCapture):
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise StreamClosedException()
        self.cur_time = 0

    def read(self) -> tuple[bool, np.ndarray, float]:
        timestamp = self.cur_time * 1000
        res, frame = self.get(timestamp)
        return res, frame, timestamp

    def get(self, timestamp) -> tuple[bool, np.ndarray]:
        self.cap.set(cv2.CAP_PROP_POS_MSEC, timestamp)
        res, frame = self.cap.read()
        return res, frame

    def set_time(self, tm):
        self.cur_time = tm

    def release(self):
        self.cap.release()


def show_board_state(image, board, val):
    cv2.imshow('image', image)
    board = board[::-1]
    val = val[::-1]
    empty = np.zeros((608, 608, 3), np.uint8)
    empty[:, :] = (181, 217, 253)
    coords = np.linspace(28, 580, 19).astype(int)
    for i in range(len(coords)):
        empty = cv2.line(empty, (coords[0], coords[i]), (coords[-1], coords[i]), (0, 0, 0), thickness=1)
        empty = cv2.line(empty, (coords[i], coords[0]), (coords[i], coords[-1]), (0, 0, 0), thickness=1)
    for i in range(3):
        for j in range(3):
            x = coords[3 + 6 * i]
            y = coords[3 + 6 * j]
            empty = cv2.circle(empty, (x, y), radius=3, color=(0, 0, 0), thickness=-1)
    for i in range(len(coords)):
        for j in range(len(coords)):
            x = coords[i]
            y = coords[j]
            xshift = 5 * len(str(val[j][i]))
            if board[j][i] == 1:
                empty = cv2.circle(empty, (x, y + 2), radius=12, color=(50, 50, 50), thickness=-1)
                empty = cv2.circle(empty, (x, y), radius=12, color=(255, 255, 255), thickness=-1)
                empty = cv2.putText(empty, str(val[j][i]), (x - xshift, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 0, 0), 1, cv2.LINE_AA)
            if board[j][i] == -1:
                empty = cv2.circle(empty, (x, y + 2), radius=12, color=(100, 100, 100), thickness=-1)
                empty = cv2.circle(empty, (x, y), radius=12, color=(0, 0, 0), thickness=-1)
                empty = cv2.putText(empty, str(val[j][i]), (x - xshift, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow('board', empty)
    key = cv2.waitKey(1)
    if key != -1:
        cv2.waitKey()


if __name__ == '__main__':
    save_path_search = '/home/kriaq/Downloads/segmentation.pth'
    save_path_detect = '/home/kriaq/Downloads/yolo8s.pt'
    video_path = '/home/kriaq/Videos/goparties/6.mp4'

    device = 'cpu'
    # if you can use cuda:
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    source = VideoCapture2(video_path)
    source.cur_time = 215
    stream_recognition = StreamRecognition(source, save_path_search, save_path_detect, device)
    game_validation = GameValidation()
    move_cnt = np.zeros((19, 19), dtype=int)
    cur_move = 1
    while True:
        start = time.time()
        try:
            print("start recognition")
            board, prob, quality, timestamp = stream_recognition.recognize()
            print("start validation")
            game_validation.validate(board, prob, quality, timestamp)
            print("end validation")
            board = game_validation.board.to_numpy()
            move = game_validation.get_move()
            source.cur_time += time.time() - start
            print(time.time() - start)
            if move is not None:
                x, y, color = move
                move_cnt[x][y] = cur_move
                cur_move += 1
            res, image = source.get(timestamp)
            print(quality, timestamp)
        except Exception as e:
            raise e
            source.cur_time += time.time() - start
            continue
        # print(prob)
        show_board_state(image, board, move_cnt)
