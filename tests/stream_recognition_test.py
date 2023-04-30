import cv2
import numpy as np
import asyncio

from src.stream_capture import VideoCapture, StreamSaver
from src.stream_recognition import StreamRecognition, StreamRecognitionProcess
from src.state_recognition.preprocessing import padding


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
    return cv2.waitKey(1)


async def task_while():
    while True:
        print('Cycle')
        await asyncio.sleep(0.2)


async def main():
    save_path_search = '../src/state_recognition/model_saves/segmentation18.pth'
    save_path_detect = '../src/state_recognition/model_saves/yolo8n.pt'
    video_path = 'video/1.mp4'

    device = 'cpu'
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    source = StreamSaver(video_path, 'save/videos/1/')
    # source.start_time -= 100
    stream_recognition = StreamRecognitionProcess(source, save_path_search, save_path_detect, device)
    stream_recognition.update_parameters(search_period=10)
    # stream_recognition.update_parameters(mode='given', points=np.array([[0, 0], [0, 1023], [1023, 1023], [1023, 0]]))
    task = asyncio.create_task(task_while())
    while True:
        try:
            board, prob, quality, timestamp, coordinates = await stream_recognition.recognize()
            last_coordinates = stream_recognition.last_coordinates()
            # res, image = source.get(timestamp)
            res, image, milliseconds = source.read()
            contour_image = cv2.drawContours(padding(image), [np.array(last_coordinates)], -1, (0, 0, 255), 2)
            print(quality, timestamp)
            # print(prob)
            key = show_board_state(contour_image, board, prob)
            if key == ord('e'):
                break
        except Exception:
            print('Predict error test')


if __name__ == '__main__':
    asyncio.run(main())
