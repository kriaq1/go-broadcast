import numpy as np
import asyncio
import multiprocessing
from multiprocessing import Queue, Array
#from ..stream_recognition import StreamRecognitionProcess
from ..stream_capture import StreamClosed, StreamImage, StreamSaver
from ..game_validation import GameValidation
from ..sgf_api import SGF
from ctypes import Structure, c_int


class Point(Structure):
    _fields_ = [('x', c_int), ('y', c_int)]


class ControllerProcess:
    def __init__(self, save_path_search: str, save_path_detect: str, device: str = 'cpu'):
        super().__init__()
        self.stream_recognition = StreamRecognitionProcess(source=StreamClosed(),
                                                           save_path_search=save_path_search,
                                                           save_path_detect=save_path_detect,
                                                           device=device)
        self.game_validation = GameValidation()
        self

    def release(self):
        pass

    async def check_parent_alive(self):
        while True:
            parent = multiprocessing.parent_process()
            if parent is None or not parent.is_alive():
                break
        self.release()

    async def run_recognition(self, queue_result_recognition):
        while True:
            # TODO
            result_recognition = await self.stream_recognition.recognize()
            if result_recognition is not None:
                queue_result_recognition.put(result_recognition)

    async def run_validation(self, queue_result_recognition, queue_result_validation):
        while True:
            # TODO
            result_recognition = queue_result_recognition.get()
            self.game_validation.validate(*result_recognition)
            result_validation = self.game_validation.get_move()
            queue_result_validation.put(result_validation)

    async def run_coordinates(self):
        while True:
            # TODO
            self.shared_coordinates = self.stream_recognition.last_coordinates()
            await asyncio.sleep(0)


async def async_run_controller(save_path_search: str, save_path_detect: str, device: str, shared_coordinates):
    controller_process = ControllerProcess(save_path_search=save_path_search,
                                           save_path_detect=save_path_detect,
                                           device=device)
    queue_result_recognition = Queue()
    queue_result_validation = Queue()
    task_alive = asyncio.create_task(controller_process.check_parent_alive())
    task_recognition = asyncio.create_task(controller_process.run_recognition(queue_result_recognition))
    task_validation = asyncio.create_task(controller_process.run_validation(queue_result_recognition,
                                                                            queue_result_validation))
    queue_update_parameters = Queue()
    while True:
        await asyncio.sleep(0)
        if not queue_update_parameters.empty():
            # TODO TRY
            parameters = queue_update_parameters.get_nowait()
            controller_process.stream_recognition.update_parameters(parameters)


def run_controller(**kwargs):
    asyncio.run(async_run_controller(**kwargs))


class Controller:
    def __init__(self, save_path_search: str, save_path_detect: str, device: str = 'cpu'):
        self.shared_coordinates = Array(Point, [(-1, -1), (-1, -1), (-1, -1), (-1, -1)])
        args = save_path_search, save_path_detect, device, self.shared_coordinates
        self.p = multiprocessing.Process(f=run_controller, args=args)
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

        self.validation_kwargs = dict()

    def update_parameters(self, **kwargs):
        for key, value in kwargs:
            self.recognition_kwargs[key] = value
        # update(self.recognition_kwargs)

    def last_coordinates(self, shape) -> np.ndarray:
        pass

    def last_board_state(self) -> np.ndarray:
        pass

    def get_sgf(self) -> SGF:
        pass

    def get_recognition_state(self, image):
        source = StreamImage(image)
        self.update_parameters(source=source)
        pass
