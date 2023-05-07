import numpy as np
import asyncio
import multiprocessing
from multiprocessing import Queue, Array, Value
from ..stream_recognition import StreamRecognitionProcess
from ..stream_capture import StreamClosed, StreamImage, StreamSaver
from ..gamelog import GameLog
from ..sgf_api import SGFAPI, SGFWriter
from ctypes import Structure, c_int
import time
import sys


class Point(Structure):
    _fields_ = [('x', c_int), ('y', c_int)]


class ControllerProcess:
    def __init__(self, save_path_search: str,
                 save_path_detect: str,
                 device: str,
                 shared_coordinates: Array,
                 shared_board_state: Array,
                 moves_queue: Queue,
                 recognition_parameters_queue: Queue,
                 gamelog_parameters_queue: Queue,
                 global_timestamp: Value
                 ):
        self.global_timestamp = global_timestamp
        self.shared_coordinates = shared_coordinates
        self.shared_board_state = shared_board_state
        self.moves_queue = moves_queue
        self.recognition_parameters_queue = recognition_parameters_queue
        self.gamelog_parameters_queue = gamelog_parameters_queue
        super().__init__()
        self.stream_recognition = StreamRecognitionProcess(source=StreamClosed(),
                                                           save_path_search=save_path_search,
                                                           save_path_detect=save_path_detect,
                                                           device=device,
                                                           global_timestamp=global_timestamp)
        self.game_log = GameLog()
        self.sgf = SGFWriter()

    def release(self):
        del self.stream_recognition
        self.sgf.save(f'./sgf-{int(time.time())}.sgf')

    async def run_recognition(self, queue_result_recognition):
        while True:
            result_recognition = await self.stream_recognition.recognize()
            if result_recognition is not None:
                state, prob, quality, timestamp, coordinates = result_recognition
                coordinates = [(p[0], p[1]) for p in coordinates]
                self.shared_coordinates[:] = coordinates
                queue_result_recognition.put((state, prob, quality, timestamp))

    async def run_validation(self, queue_result_recognition):
        while True:
            try:
                result_recognition = queue_result_recognition.get_nowait()
                self.game_log.append_state(*result_recognition)
            except Exception:
                pass
            await asyncio.sleep(0)

    async def run_getting_moves(self):
        while True:
            move = await self.game_log.get_move()
            if move is None:
                continue
            self.sgf.add(move.y + 1, move.x + 1, move.color)
            self.sgf.save(path=f'./sgf.sgf')
            self.moves_queue.put(move)
            state, val = self.game_log.get_state()
            self.shared_board_state[:] = np.resize(state, 19 * 19)


async def async_run_controller(save_path_search: str,
                               save_path_detect: str,
                               device,
                               shared_coordinates: Array,
                               shared_board_state: Array,
                               moves_queue: Queue,
                               recognition_parameters_queue: Queue,
                               gamelog_parameters_queue: Queue,
                               global_timestamp: Value):
    controller_process = ControllerProcess(save_path_search=save_path_search,
                                           save_path_detect=save_path_detect,
                                           device=device,
                                           shared_coordinates=shared_coordinates,
                                           shared_board_state=shared_board_state,
                                           moves_queue=moves_queue,
                                           recognition_parameters_queue=recognition_parameters_queue,
                                           gamelog_parameters_queue=gamelog_parameters_queue,
                                           global_timestamp=global_timestamp)
    queue_result_recognition = Queue()

    task_recognition = asyncio.create_task(controller_process.run_recognition(queue_result_recognition))
    task_validation = asyncio.create_task(controller_process.run_validation(queue_result_recognition))
    task_getting_moves = asyncio.create_task(controller_process.run_getting_moves())
    while True:
        await asyncio.sleep(0)
        while not recognition_parameters_queue.empty():
            try:
                parameters = recognition_parameters_queue.get_nowait()
                controller_process.stream_recognition.update_parameters(**parameters)
            except Exception:
                pass
        while not gamelog_parameters_queue.empty():
            try:
                parameters = gamelog_parameters_queue.get_nowait()
            except Exception:
                pass

        parent = multiprocessing.parent_process()
        if parent is None or not parent.is_alive():
            break
    task_recognition.cancel()
    task_validation.cancel()
    task_getting_moves.cancel()


def run_controller(**kwargs):
    asyncio.run(async_run_controller(**kwargs))
