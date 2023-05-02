import numpy as np
import asyncio
import multiprocessing
from multiprocessing import Queue, Array
from ..stream_recognition import StreamRecognitionProcess
from ..stream_capture import StreamClosed, StreamImage, StreamSaver
from ..game_validation import GameValidation
from ..sgf_api import SGFAPI
from ctypes import Structure, c_int


class GameLog:
    async def approve(self, state: np.ndarray, prob: np.ndarray, timestamp: float):
        pass

    async def get_move(self):
        pass


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
        self.sgf_api = SGF()

    def release(self):
        del self.stream_recognition
        self.sgf_api.save()

    async def check_parent_alive(self):

        while True:
            parent = multiprocessing.parent_process()
            if parent is None or not parent.is_alive():
                break
            await asyncio.sleep(0)
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


async def async_run_controller(save_path_search: str,
                               save_path_detect: str,
                               device,
                               shared_coordinates: Array,
                               shared_board_state: Array,
                               moves_queue: Queue,
                               recognition_parameters_queue: Queue,
                               gamelog_parameters_queue: Queue):
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
