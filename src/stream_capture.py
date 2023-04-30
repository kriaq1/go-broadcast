import cv2
import os
from pathlib import Path
import numpy as np
import time
from multiprocessing import Process, Lock, Value, shared_memory

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

    def __del__(self):
        self.release()


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


def create_shared_memory_nparray(data, name):
    release_shared(name)
    np_type = data.dtype
    np_shape = data.shape
    d_size = np.dtype(np_type).itemsize * np.prod(np_shape)
    shm = shared_memory.SharedMemory(create=True, size=d_size, name=name)
    dst = np.ndarray(shape=np_shape, dtype=np_type, buffer=shm.buf)
    dst[:] = data[:]
    return shm


def release_shared(name):
    try:
        shm = shared_memory.SharedMemory(name=name)
        shm.close()
        shm.unlink()
    except Exception:
        pass


def save_frame(save_path, timestamp, frame) -> None:
    cv2.imwrite(save_path + str(timestamp) + '.jpg', frame)


def get_frame(save_path, timestamp) -> tuple[bool, np.ndarray | None]:
    try:
        files = os.listdir(save_path)
        files = [file for file in files if file.endswith('.jpg') and file[:-4].isdigit()]
        timestamps = [int(file[:-4]) for file in files]
        file = files[np.argmin(np.abs(np.array(timestamps) - timestamp))]
        return True, cv2.imread(save_path + file)
    except Exception:
        return False, None


def release_frames(save_path):
    files = [file for file in os.listdir(save_path) if file.endswith('.jpg') and file[:-4].isdigit()]
    for file in files:
        os.remove(save_path + file)


def get_percentage_timestamp(save_path, percentage):
    try:
        assert 0 <= percentage <= 100
        files = os.listdir(save_path)
        files = [file for file in files if file.endswith('.jpg') and file[:-4].isdigit()]
        timestamps = [int(file[:-4]) for file in files]
        # timestamps = list(sorted(timestamps))
        # return timestamps[int((len(timestamps) - 1) * percentage / 100)]
        min_timestamp = min(timestamps)
        max_timestamp = max(timestamps)
        return int((max_timestamp - min_timestamp) * percentage / 100 + min_timestamp)
    except Exception:
        return 0


def read_and_save_source(lock, frame_count, shared_ndarray, source, save_path, start_timestamp, fps_save, fps_update):
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_timestamp_save = start_timestamp
    start_timestamp_update = start_timestamp
    fps_save = min(fps_save, fps_update)
    start_time = time.time()
    while True:
        res, frame = cap.read()
        if not res:
            break
        new_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        if save_path is not None and new_timestamp - start_timestamp_save >= 1000 / fps_save:
            save_frame(save_path, int(new_timestamp), frame)
            start_timestamp_save = new_timestamp
        if new_timestamp - start_timestamp_update >= 1000 / fps_update:
            lock.acquire()
            shared_ndarray[:] = frame[:]
            frame_count.value = int(new_timestamp)
            lock.release()
            start_timestamp_update = new_timestamp
        if (time.time() - start_time) * 1000 < new_timestamp - start_timestamp - 500:
            time.sleep(1 / fps)

    cap.release()


class StreamSaver(StreamCapture):
    def __init__(self, source, save_path=None, fps_save=0.1, fps_update=20, shared_name='ndarray'):
        cap = cv2.VideoCapture(source)
        res, frame = cap.read()
        if not cap.isOpened() or not res:
            raise StreamClosedException()
        start = cap.get(cv2.CAP_PROP_POS_MSEC)
        cap.release()
        if save_path is not None:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            save_frame(save_path, int(start), frame)
        self.save_path = save_path
        self.fps_save = fps_save
        self.fps_update = fps_update
        self.lock = Lock()
        self.frame_count = Value('i', 0)
        self.shared_name = shared_name
        self.shared_buffer = create_shared_memory_nparray(frame, shared_name)
        self.shared_ndarray = np.ndarray(frame.shape, dtype=frame.dtype, buffer=self.shared_buffer.buf)
        args = (self.lock, self.frame_count, self.shared_ndarray, source, save_path, start, fps_save, fps_update)
        self.p = Process(target=read_and_save_source, args=args)
        self.p.start()

    def read(self) -> tuple[bool, np.ndarray, float]:
        if self.p.is_alive():
            self.lock.acquire()
            res, frame, timestamp = True, self.shared_ndarray.copy(), self.frame_count.value
            self.lock.release()
        else:
            res, frame, timestamp = False, None, None
        return res, frame, timestamp

    def get(self, timestamp) -> tuple[bool, np.ndarray | None]:
        if self.save_path is None:
            return self.read()[:2]
        return get_frame(self.save_path, timestamp)

    def release(self):
        self.p.terminate()
        self.p.join()
        release_shared(self.shared_name)
        if self.save_path is not None:
            release_frames(self.save_path)
            self.save_path = None


def available_camera_indexes_list(max_index=10):
    available_result = []
    for index in range(max_index + 1):
        cap = cv2.VideoCapture(index)
        if cap.isOpened() and cap.read()[0]:
            available_result.append(index)
        cap.release()
    return available_result
