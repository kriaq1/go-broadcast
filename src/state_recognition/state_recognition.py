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
                  conf: float = 0.4,
                  iou: float = 0.5,
                  min_distance: float = 0,
                  max_distance: float = 1, ) -> tuple[np.ndarray, np.ndarray, float]:
        image = padding(image, 1024, inter=cv2.INTER_CUBIC)
        if mode == 'prev' and self.coordinates is not None:
            coordinates = self.coordinates
        elif mode == 'given' and points is not None:
            coordinates = points
            self.segmentation_quality = 1
        else:
            mask = self.search.get_mask_image(image)
            coordinates = self.search.find_quadrilateral(mask)
            segmentation_quality = get_segmentation_quality(mask, coordinates)
        cut = linear_perspective(image, coordinates, 608)
        results = self.stone_detector.get_predict(source=cut, conf=conf, iou=iou, max_det=1000)
        board_state, probabilities = get_board_state(results[0], min_distance, max_distance)
        return board_state, probabilities, segmentation_quality


def get_segmentation_quality(mask, coordinates):
    return 1
