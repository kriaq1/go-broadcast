import cv2
import numpy as np
import asyncio

from src.stream_capture import VideoCapture, StreamSaver
from src.stream_recognition import StreamRecognition


def show_board_state(image, board, probabilities):
    cv2.imshow('image', image)
    board = board.copy()
    prob = probabilities.copy()
    empty = np.zeros((608, 608, 3), np.uint8)
    empty[:, :] = (181, 217, 253)
    coords = np.linspace(28, 580, 19).astype(int)
    for i in range(len(coords)):
        cv2.line(empty, (coords[0], coords[i]), (coords[-1], coords[i]), (0, 0, 0), thickness=1)
        cv2.line(empty, (coords[i], coords[0]), (coords[i], coords[-1]), (0, 0, 0), thickness=1)
    for i in range(len(coords)):
        for j in range(len(coords)):
            x = coords[i]
            y = coords[j]
            if prob[j][i] == 0:
                cv2.circle(empty, (x, y), radius=12, color=(0, 0, 255), thickness=-1)
                continue
            if board[j][i] == 1:
                cv2.circle(empty, (x, y), radius=12, color=(255, 255, 255), thickness=-1)
            if board[j][i] == -1:
                cv2.circle(empty, (x, y), radius=12, color=(0, 0, 0), thickness=-1)
    cv2.imshow('board', empty)
    cv2.waitKey()


async def main():
    save_path_search = '../src/state_recognition/model_saves/segmentation18.pth'
    save_path_detect = '../src/state_recognition/model_saves/yolo8n.pt'
    video_path = 'video/1.mp4'

    device = 'cpu'
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    source = StreamSaver(video_path)
    # source.start_time -= 100
    stream_recognition = StreamRecognition(source, save_path_search, save_path_detect, device)

    while True:
        try:
            board, prob, quality, timestamp, coordinates = await stream_recognition.recognize()
            res, image = source.get(timestamp)
            print(quality, timestamp)
            # print(prob)
            show_board_state(image, board, prob)
        except Exception:
            print('Predict error')

if __name__ == '__main__':
    asyncio.run(main())
