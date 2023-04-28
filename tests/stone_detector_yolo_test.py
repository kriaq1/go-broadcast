from os import listdir

import cv2
import torch

from src.state_recognition.feature_extractor import linear_perspective

from src.state_recognition.find_board_nn import BoardSearch
from src.state_recognition.find_board_nn import load_image

from src.state_recognition.detect_stones_yolo import StoneDetector


if __name__ == '__main__':
    save_path_search = '../src/state_recognition/model_saves/segmentation18.pth'
    save_path_detect = '../src/state_recognition/model_saves/yolo8n.pt'
    test_path = 'images/test/'
    result_path = 'images/result_detect/'

    device = torch.device('cpu')
    # if you can use cuda:
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    search = BoardSearch(device=device, save_path=save_path_search)
    stone_detector = StoneDetector(device=device, save_path=save_path_detect)

    for file in sorted(listdir(test_path)):
        image = load_image(test_path + str(file))
        points = search.get_board_coordinates(image)
        cut = linear_perspective(image, points, 608)
        results = stone_detector.get_predict(source=cut, conf=0.4, iou=0.5, max_det=1000)
        annotated_frame = results[0].plot(line_width=1)
        boxes = results[0].boxes.xywh
        boxes_n = results[0].boxes.xywhn
        conf = results[0].boxes.conf
        cls = results[0].boxes.cls
        data = results[0].boxes.data

        # print(boxes.shape, boxes_n.shape, conf.shape, cls.shape, data.shape)

        cv2.imwrite(result_path + file, annotated_frame)
