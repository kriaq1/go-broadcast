import torch
import numpy as np
import asyncio
import sys
import multiprocessing
from multiprocessing import Process, Queue, Array
from ctypes import Structure, c_int

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
        self.quality_coeff = 0.975
        self.conf = 0.4
        self.iou = 0.5
        self.min_distance = 20 / 608
        self.max_distance = 50 / 608
        self.search_period = 4
        self.predict_epoch = 0
        self.mode = None
        self.points = None
        try:
            self.state_recognition = StateRecognition(save_path_search, save_path_detect, torch.device(device))
        except Exception:
            raise LoadError

    def recognize(self) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
        res, frame, timestamp = self.source.read()
        if not res:
            raise StreamReadError
        mode = self.mode
        points = self.points
        if mode is None and self.predict_epoch % self.search_period != 0:
            mode = 'prev'
        try:
            parameters = dict(image=frame, mode=mode, points=points, conf=self.conf, iou=self.iou,
                              min_distance=self.min_distance, max_distance=self.max_distance,
                              segmentation_threshold=self.seg_threshold, quality_coefficient=self.quality_coeff)
            board, prob, quality = self.state_recognition.get_board(**parameters)
        except Exception:
            raise PredictError
        self.predict_epoch += 1
        return board, prob, quality, timestamp, self.state_recognition.coordinates

    def last_coordinates(self):
        return self.state_recognition.coordinates

    def update_parameters(self,
                          source: StreamCapture = None,
                          mode: str = None,
                          points: np.ndarray = None,
                          search_period: int = 4,
                          seg_threshold: float = 0.5,
                          quality_coefficient: float = 0.975,
                          conf: float = 0.4,
                          iou: float = 0.5,
                          min_distance: float = 20 / 608,
                          max_distance: float = 50 / 608):

        if source is not None:
            self.source = source
        self.mode = mode
        self.points = points
        self.seg_threshold = seg_threshold
        self.quality_coeff = quality_coefficient
        self.conf = conf
        self.iou = iou
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.search_period = search_period
        self.predict_epoch = 0
        self.state_recognition.coordinates = None


class Point(Structure):
    _fields_ = [('x', c_int), ('y', c_int)]


def run_stream_recognition(queue_recognize, queue_update_parameters, coordinates, init_parameters):
    try:
        stream_recognition = StreamRecognition(**init_parameters)
    except LoadError:
        print('Load Error', file=sys.stderr)
        queue_recognize.close()
        return
    while True:
        try:
            result = stream_recognition.recognize()
            queue_recognize.put(result)
        except StreamReadError:
            pass
        except PredictError:
            print('Predict Error', file=sys.stderr)
            pass
        except Exception:
            print('Recognize Error', file=sys.stderr)
            break

        parent = multiprocessing.parent_process()
        if parent is None or not parent.is_alive():
            break

        try:
            with coordinates.get_lock():
                for coordinate, last_coordinate in zip(coordinates, stream_recognition.last_coordinates()):
                    coordinate.x, coordinate.y = last_coordinate[0], last_coordinate[1]
        except Exception:
            pass

        while not queue_update_parameters.empty():
            try:
                parameters = queue_update_parameters.get_nowait()
                stream_recognition.update_parameters(**parameters)
            except Exception:
                break
    queue_recognize.close()


class StreamRecognitionProcess:
    def __init__(self,
                 source: StreamCapture,
                 save_path_search: str,
                 save_path_detect: str,
                 device: str = 'cpu', ):
        init = dict(source=source, save_path_search=save_path_search, save_path_detect=save_path_detect, device=device)
        self.queue_recognize = Queue()
        self.queue_update_parameters = Queue()
        self.coordinates = Array(Point, [(0, 0), (0, 0), (0, 0), (0, 0)])
        args = (self.queue_recognize, self.queue_update_parameters, self.coordinates, init)
        self.p = Process(target=run_stream_recognition, args=args)
        self.p.start()

    async def recognize(self) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray] | None:
        while self.p.is_alive() and self.queue_recognize.empty():
            await asyncio.sleep(0)
        try:
            return self.queue_recognize.get_nowait()
        except Exception:
            return None

    def last_coordinates(self) -> np.ndarray:
        result = np.zeros((4, 2), dtype=int)
        try:
            with self.coordinates.get_lock():
                for i, coordinate in enumerate(self.coordinates):
                    result[i] = coordinate.x, coordinate.y
        except Exception:
            result = np.array([[0, 0], [0, 1023], [1023, 1023], [1023, 0]])
        return result

    def update_parameters(self,
                          source: StreamCapture = None,
                          mode: str = None,
                          points: np.ndarray = None,
                          search_period: int = 4,
                          seg_threshold: float = 0.5,
                          quality_coefficient: float = 0.975,
                          conf: float = 0.4,
                          iou: float = 0.5,
                          min_distance: float = 20 / 608,
                          max_distance: float = 50 / 608):
        parameters = dict(source=source, mode=mode, points=points, search_period=search_period,
                          seg_threshold=seg_threshold, quality_coefficient=quality_coefficient,
                          conf=conf, iou=iou, min_distance=min_distance, max_distance=max_distance)
        self.queue_update_parameters.put(parameters)

    def empty(self):
        return self.queue_recognize.empty()

    def is_alive(self):
        return self.p.is_alive()

    def __del__(self):
        self.queue_update_parameters.close()
        self.p.terminate()
        self.p.join()
