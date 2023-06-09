import numpy as np
import multiprocessing
from multiprocessing import Queue, Array
from ..stream_capture import StreamClosed, StreamImage, StreamSaver
from ..sgf_api import SGFAPI
from ctypes import Structure, c_int
from .process_controller import run_controller
from . import utils


class Point(Structure):
    _fields_ = [('x', c_int), ('y', c_int)]


class Controller:
    def __init__(self, save_path_search: str, save_path_detect: str, save_path_sgf: str, device: str, global_timestamp):
        self.global_timestamp = global_timestamp
        self.shared_coordinates: Array = Array(Point, [(-1, -1), (-1, -1), (-1, -1), (-1, -1)])
        self.shared_board_state: Array = Array('i', [0] * 19 * 19)
        self.shared_values: Array = Array('i', [0] * 19 * 19)
        self.moves_queue: Queue = Queue()
        self.recognition_parameters_queue: Queue = Queue()
        self.gamelog_parameters_queue: Queue = Queue()
        self.padded_size = (1024, 1024)
        self.save_path_sgf = save_path_sgf
        kwargs = dict(save_path_search=save_path_search,
                      save_path_detect=save_path_detect,
                      save_path_sgf=save_path_sgf,
                      device=device,
                      shared_coordinates=self.shared_coordinates,
                      shared_board_state=self.shared_board_state,
                      shared_values=self.shared_values,
                      moves_queue=self.moves_queue,
                      recognition_parameters_queue=self.recognition_parameters_queue,
                      gamelog_parameters_queue=self.gamelog_parameters_queue,
                      global_timestamp=global_timestamp)

        self.p = multiprocessing.Process(target=run_controller, kwargs=kwargs)
        self.p.start()
        self.recognition_kwargs = dict(source=None,
                                       mode=None,
                                       points=None,
                                       search_period=4,
                                       seg_threshold=0.5,
                                       quality_coefficient=0.975,
                                       conf=0.4,
                                       iou=0.5,
                                       min_distance=20 / 608,
                                       max_distance=50 / 608)

        self.gamelog_kwargs = dict(delay=1,
                                   appearance_count=3,
                                   valid_time=3 * 1000,
                                   valid_zeros_count=20,
                                   valid_thresh=0.2)

    def update_recognition_parameters(self, **kwargs):
        for key in kwargs.keys():
            self.recognition_kwargs[key] = kwargs[key]
        if self.recognition_kwargs['source'] is not None:
            self.recognition_kwargs['source'] = self.recognition_kwargs['source'].copy()
        self.recognition_parameters_queue.put(self.recognition_kwargs)

    def update_gamelog_parameters(self, **kwargs):
        for key in kwargs.keys():
            self.gamelog_kwargs[key] = kwargs[key]
        self.gamelog_parameters_queue.put(self.gamelog_kwargs)

    def last_coordinates(self, shape) -> np.ndarray:
        points = np.array([[p.x, p.y] for p in self.shared_coordinates[:]], dtype=int)
        return utils.unpadding_points(points, shape=shape, padded_shape=self.padded_size)

    def last_board_state(self) -> tuple[np.ndarray, np.ndarray | None]:
        return np.resize(np.array(self.shared_board_state), (19, 19)), np.resize(np.array(self.shared_values), (19, 19))

    def get_sgf(self) -> SGFAPI:
        pass

    def get_recognition_state(self, image):
        source = StreamImage(image)
        self.update_recognition_parameters(source=source)

    def clear_validation(self):
        kwargs = self.gamelog_kwargs.copy()
        kwargs['delay'] = 0
        kwargs.update(initial_state=np.zeros((19, 19), dtype=int))
        self.gamelog_parameters_queue.put(kwargs)

    def __del__(self):
        self.p.terminate()
        self.p.join()
