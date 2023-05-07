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
    def __init__(self,
                 save_path_search: str,
                 save_path_detect: str,
                 save_path_sgf: str,
                 device: str,
                 shared_coordinates: Array,
                 shared_board_state: Array,
                 moves_queue: Queue,
                 recognition_parameters_queue: Queue,
                 gamelog_parameters_queue: Queue,
                 global_timestamp: Value):
        self.save_path_sgf = save_path_sgf
        self.global_timestamp = global_timestamp
        self.shared_coordinates = shared_coordinates
        self.shared_board_state = shared_board_state
        self.moves_queue = moves_queue
        self.recognition_parameters_queue = recognition_parameters_queue
        self.gamelog_parameters_queue = gamelog_parameters_queue
        self.stream_recognition = StreamRecognitionProcess(source=StreamClosed(),
                                                           save_path_search=save_path_search,
                                                           save_path_detect=save_path_detect,
                                                           device=device,
                                                           global_timestamp=global_timestamp)
        self.game_log = GameLog()
        self.sgf = SGFWriter()

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

    async def run_get_moves(self):
        while True:
            move = await self.game_log.get_move()
            if move is None:
                continue
            try:
                self.sgf.add(move.x, move.y, move.color)
            except Exception:
                print('SGF Error', file=sys.stderr)
            self.sgf.save(path=self.save_path_sgf)
            self.moves_queue.put(move)
            state, val = self.game_log.get_state()
            self.shared_board_state[:] = np.resize(state, 19 * 19)

    async def run_recognition_parameters(self):
        while True:
            while not self.recognition_parameters_queue.empty():
                try:
                    parameters = self.recognition_parameters_queue.get_nowait()
                    self.stream_recognition.update_parameters(**parameters)
                except Exception:
                    pass
            await asyncio.sleep(0)

    async def run_validation_parameters(self):
        while True:
            while not self.gamelog_parameters_queue.empty():
                try:
                    parameters = self.gamelog_parameters_queue.get_nowait()
                    if 'initial_state' in parameters.keys():
                        self.sgf = SGFWriter()
                        self.shared_board_state[:] = np.zeros(19 * 19, dtype=int)
                    self.game_log.update_parameters(**parameters)
                except Exception:
                    pass
            await asyncio.sleep(0)

    def release(self):
        del self.stream_recognition


async def async_run_controller(**kwargs):
    controller_process = ControllerProcess(**kwargs)

    queue_result_recognition = Queue()

    task_recognition = asyncio.create_task(controller_process.run_recognition(queue_result_recognition))
    task_validation = asyncio.create_task(controller_process.run_validation(queue_result_recognition))
    task_get_moves = asyncio.create_task(controller_process.run_get_moves())
    task_recognition_parameters = asyncio.create_task(controller_process.run_recognition_parameters())
    task_validation_parameters = asyncio.create_task(controller_process.run_validation_parameters())

    while True:
        parent = multiprocessing.parent_process()
        if parent is None or not parent.is_alive():
            break
        await asyncio.sleep(1)

    task_recognition.cancel()
    task_validation.cancel()
    task_get_moves.cancel()
    task_recognition_parameters.cancel()
    task_validation_parameters.cancel()
    controller_process.release()


def run_controller(**kwargs):
    asyncio.run(async_run_controller(**kwargs))
