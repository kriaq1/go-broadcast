from .gamelog import GameLog
from .broadcast import Broadcast
from .board import Board, Turn
from .state_recognition import StateRecognition
from .api import API

import cv2 as cv

import asyncio


def trace(f):
    return f
    async def wrapper(*args):
        print("Calling", f.__name__)
        return f(*args)
    return wrapper


def loop(f):
    async def wrapper(*args):
        while True:
            if not await f(*args):
                return
    return wrapper


class DataLine:
    def __init__(self, comps: list, producers: list, consumers: list):
        assert len(comps) == len(producers) + 1
        assert len(producers) == len(consumers)
        self.comps = comps
        self.producers = producers
        self.consumers = consumers

    def run(self):
        queues = [asyncio.Queue() for i in range(len(self.producers))]
        ioloop = asyncio.get_event_loop()

        def make_producer(f, comp):
            async def prod(q):
                while True:
                    x = f(comp)
                    await q.put(x)
            return prod

        def make_consumer(f, comp):
            async def cons(q):
                while True:
                    x = await q.get()
                    f(comp, x)
            return cons

        producers_tasks = [
            ioloop.create_task(make_producer(prod, comp))
            for (prod, comp, q)
            in zip(self.producers, self.comps[:-1], queues)
        ]

        consumers_tasks = [
            ioloop.create_task(make_consumer(cons, comp))
            for (cons, comp, q)
            in zip(self.consumers, self.comps[1:], queues)
        ]

        waits = asyncio.wait(producers_tasks + consumers_tasks)
        ioloop.run_forever(waits)


class App:
    __slots__ = ["source", "recognition", "gamelog", "broadcast"]

    def __init__(self, source: cv.VideoCapture, api: list[API], *recognition_args):
        self.source = source
        self.gamelog: GameLog = GameLog()
        self.broadcast: Broadcast = Broadcast(api)
        self.recognition = StateRecognition(*recognition_args)

    def start(self):
        @loop
        @trace
        async def scan(scan_edit_queue):
            ret, frame = self.source.read()
            if not ret:
                return False
            b = self.recognition.get_board(frame)
            if not b:
                return False
            print("scan: got board")
            await scan_edit_queue.put(b[0])
            return True

        @loop
        @trace
        async def edit(scan_edit_queue, edit_broadcast_queue):
            b = await scan_edit_queue.get()
            print("edit: edited board")
            self.editor.add(b)
            await edit_broadcast_queue.put(self.editor.get())
            scan_edit_queue.task_done()
            return True

        @loop
        @trace
        async def broadcast(edit_broadcast_queue):
            b = await edit_broadcast_queue.get()
            for api in self.api:
                try:
                    api.add(b)
                    api.broadcast()
                except Exception:
                    pass
            print("broadcast: broadcasted board")
            edit_broadcast_queue.task_done()
            return True

        sc_ed = asyncio.Queue()
        ed_br = asyncio.Queue()

        ioloop = asyncio.get_event_loop()

        br = ioloop.create_task(broadcast(ed_br))
        ed = ioloop.create_task(edit(sc_ed, ed_br))
        sc = ioloop.create_task(scan(sc_ed))

        wait_tasks = asyncio.wait([br, ed, sc])
        ioloop.run_until_complete(wait_tasks)
        ioloop.close()
