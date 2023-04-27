import torch
import numpy as np

from .state_recognition.state_recognition import StateRecognition
from .stream_capture import StreamCapture


class RecognitionError(Exception):
    pass


class LoadError(RecognitionError):
    pass


class StreamReadError(RecognitionError):
    pass


class PredictError(RecognitionError):
    pass


class StreamRecognition:
    def __init__(self,
                 source: StreamCapture,
                 save_path_search: str,
                 save_path_detect: str,
                 device: str = 'cpu', ):
        self.source = source
        self.seg_threshold = 0.5
        self.quality_coefficient = 0.975
        self.conf = 0.4
        self.iou = 0.5
        self.min_distance = 20 / 608
        self.max_distance = 50 / 608
        self.search_period = 4
        self.predict_epoch = 0
        try:
            self.state_recognition = StateRecognition(save_path_search, save_path_detect, torch.device(device))
        except Exception:
            raise LoadError

    def recognize(self, mode: str = None, points: np.ndarray = None) -> tuple[
        np.ndarray, np.ndarray, float, float, np.ndarray]:
        res, frame, timestamp = self.source.read()
        if not res:
            raise StreamReadError
        if mode is None and self.predict_epoch % self.search_period != 0:
            mode = 'prev'
        try:
            parameters = dict(image=frame, mode=mode, points=points, conf=self.conf, iou=self.iou,
                              min_distance=self.min_distance, max_distance=self.max_distance,
                              segmentation_threshold=self.seg_threshold, quality_coefficient=self.quality_coefficient)
            board, prob, quality = self.state_recognition.get_board(**parameters)
        except Exception:
            raise PredictError
        self.predict_epoch += 1
        return board, prob, quality, timestamp, self.state_recognition.coordinates

    def update_parameters(self,
                          source: StreamCapture = None,
                          search_period: int = 4,
                          seg_threshold: float = 0.5,
                          quality_coefficient: float = 0.975,
                          conf: float = 0.4,
                          iou: float = 0.5,
                          min_distance: float = 20 / 608,
                          max_distance: float = 50 / 608):
        if source is not None:
            self.source = source
        self.seg_threshold = seg_threshold
        self.quality_coefficient = quality_coefficient
        self.conf = conf
        self.iou = iou
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.search_period = search_period
        self.predict_epoch = 0
        self.state_recognition.coordinates = None
