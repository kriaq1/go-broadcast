from .gamelog import GameLog
from .broadcast import Broadcast
from .board import Board, Turn
from .state_recognition import StateRecognition
from .api import API

import asyncio


def trace(f):
    async def wrapper(*args):
        print("Calling", f.__name__)
        await f(*args)
    return wrapper


def loop(f):
    async def wrapper(*args):
        while True:
            if not await f(*args):
                return
    return wrapper


class App:
    __slots__ = ["recognition", "gamelog", "broadcast"]

    def __init__(self, api: list[API], *recognition_args):
        self.gamelog: GameLog = GameLog()
        self.broadcast: Broadcast = Broadcast(api)
        self.recognition = StateRecognition(*recognition_args)

    def start(self):
        @loop
        @trace
        async def scan(scan_edit_queue):
            b = await self.recognition.get_board()
            if not b:
                return False
            print("scan: got board")
            await scan_edit_queue.put(b)
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
