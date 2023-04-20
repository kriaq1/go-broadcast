import cv2
import numpy as np
import time

from abc import ABC, abstractmethod


class StreamCaptureException(Exception):
    pass


class StreamClosedException(StreamCaptureException):
    pass


class StreamCapture(ABC):
    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray, float]:
        pass

    @abstractmethod
    def get(self, timestamp) -> tuple[bool, np.ndarray]:
        pass

    @abstractmethod
    def release(self):
        pass


class VideoCapture(StreamCapture):
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise StreamClosedException()
        self.start_time = time.time()

    def read(self) -> tuple[bool, np.ndarray, float]:
        timestamp = (time.time() - self.start_time) * 1000
        res, frame = self.get(timestamp)
        return res, frame, timestamp

    def get(self, timestamp) -> tuple[bool, np.ndarray]:
        self.cap.set(cv2.CAP_PROP_POS_MSEC, timestamp)
        res, frame = self.cap.read()
        return res, frame

    def release(self):
        self.cap.release()
