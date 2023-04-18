import cv2
import numpy as np

from .detect_stones_yolo import StoneDetector
from .find_board_nn import BoardSearch
from .board_state import get_board_state

from .feature_extractor import linear_perspective
from .preprocessing import padding


class StateRecognition:
    def __init__(self, save_path_search: str, save_path_detect: str, device):
        self.search = BoardSearch(device=device, save_path=save_path_search)
        self.stone_detector = StoneDetector(device=device, save_path=save_path_detect)
        self.coordinates = None
        self.segmentation_quality = 1

    def get_board(self,
                  image: np.ndarray,
                  mode: str = None,
                  points: np.ndarray = None,
                  segmentation_threshold=0.5,
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
            self.coordinates = self.search.find_quadrilateral(mask)
            self.segmentation_quality = get_segmentation_quality(mask, self.coordinates)
        cut = linear_perspective(image, self.coordinates, 608)
        results = self.stone_detector.get_predict(source=cut, conf=conf, iou=iou, max_det=1000)
        board_state, probabilities = get_board_state(results[0], min_distance, max_distance)
        return board_state, probabilities, self.segmentation_quality


def get_segmentation_quality(mask, coordinates, min_area=100):
    points = np.array(coordinates)
    image = np.zeros(mask.shape[:2])
    square_mask = cv2.fillPoly(image, pts=[points], color=1)
    mask = mask / 255
    intersection = 2 * np.sum(mask * square_mask)
    area1 = np.sum(mask)
    area2 = np.sum(square_mask)
    sum_area = area1 + area2
    if area1 < min_area:
        return 0
    return intersection / sum_area
