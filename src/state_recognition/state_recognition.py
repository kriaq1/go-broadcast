import cv2
import numpy as np

from .detect_stones_yolo import StoneDetector
from .find_board_nn import BoardSearch
from .board_state import get_board_state

from .feature_extractor import linear_perspective
from .preprocessing import padding

import math


class StateRecognition:
    def __init__(self, save_path_search: str, save_path_detect: str, device):
        self.search = BoardSearch(device=device, save_path=save_path_search)
        self.stone_detector = StoneDetector(device=device, save_path=save_path_detect)
        self.coordinates = None
        self.segmentation_quality = 0

    def get_board(self,
                  image: np.ndarray,
                  mode: str = None,
                  points: np.ndarray = None,
                  segmentation_threshold=0.5,
                  quality_coefficient=0.95,
                  conf: float = 0.4,
                  iou: float = 0.5,
                  min_distance: float = 20 / 608,
                  max_distance: float = 50 / 608, ) -> tuple[np.ndarray, np.ndarray, float]:
        image = padding(image, 1024, inter=cv2.INTER_CUBIC)
        if mode == 'prev' and self.coordinates is not None and self.segmentation_quality > 0:
            pass
        elif mode == 'given' and points is not None:
            self.coordinates = points
            self.segmentation_quality = 1
        else:
            mask = self.search.get_mask_image(image, out_threshold=segmentation_threshold)
            if np.sum(mask) / 255 < 361:
                coordinates = np.array([[0, 0], [0, 1024], [1024, 1024], [1024, 0]])
                segmentation_quality = 0
            else:
                coordinates = self.search.find_quadrilateral(mask)
                segmentation_quality = get_segmentation_quality(mask, coordinates)
            if self.coordinates is None or segmentation_quality >= self.segmentation_quality * quality_coefficient:
                self.coordinates = coordinates
                self.segmentation_quality = segmentation_quality
        cut = linear_perspective(image, self.coordinates, 608)
        results = self.stone_detector.get_predict(source=cut, conf=conf, iou=iou, max_det=500)
        board_state, probabilities = get_board_state(results[0], min_distance, max_distance)
        return board_state, probabilities, self.segmentation_quality


def get_sin(a, b, c):
    ang = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    return math.sin(ang)


def get_angle_quality(coordinates):
    coordinates = coordinates.tolist()
    sin1 = get_sin(coordinates[0], coordinates[1], coordinates[2])
    sin2 = get_sin(coordinates[1], coordinates[2], coordinates[3])
    sin3 = get_sin(coordinates[2], coordinates[3], coordinates[0])
    sin4 = get_sin(coordinates[3], coordinates[0], coordinates[1])
    return min(sin1, sin2, sin3, sin4)


def get_segmentation_quality(mask, coordinates, min_area=361):
    points = np.array(coordinates)
    square_mask = np.zeros(mask.shape[:2])
    cv2.fillPoly(square_mask, pts=[points], color=1)
    mask = mask / 255
    intersection = 2 * np.sum(mask * square_mask)
    area1 = np.sum(mask)
    area2 = np.sum(square_mask)
    sum_area = area1 + area2
    if area1 < min_area:
        return 0
    angle_quality = get_angle_quality(coordinates)
    return 0.5 * intersection / sum_area + 0.5 * angle_quality
