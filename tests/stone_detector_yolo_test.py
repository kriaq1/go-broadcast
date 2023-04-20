import matplotlib.pyplot as plt
from os import listdir

import cv2
import torch
import numpy as np

from src.feature_extractor import linear_perspective

from src.find_board_nn import BoardSearch
from src.find_board_nn import load_image

from src.detect_stones_yolo import StoneDetector


if __name__ == '__main__':
    save_path_search = '../configs/model_saves/segmentation.pth'
    save_path_detect = '../configs/model_saves/yolo8m.pt'
    test_path = 'predict_images/original/'
    result_cut_path = 'predict_images/cut/'
    result_predict_path = 'predict_images/predict/'

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

        print(boxes.shape, boxes_n.shape, conf.shape, cls.shape, data.shape)

        cv2.imshow('annotated frame', annotated_frame)
        cv2.waitKey()
