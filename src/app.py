from gamelog import GameLog
from broadcast import Broadcast
from board_scanner import BoardScanner
from api import API

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


class DataLine:
    def __init__(self, components: list, queue_type: list):
        assert len(components) = len(queue_type) + 1

class App:
    __slots__ = ["recognition", "gamelog", "broadcast"]

    def __init__(self, api: list[API], *recognition_args):
        self.gamelog: GameLog = GameLog()
        self.broadcast: Broadcast = Broadcast(api)
        self.recognition = BoardScanner(*recognition_args)

    async def start(self):
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

        br = asyncio.create_task(broadcast(ed_br))
        ed = asyncio.create_task(edit(sc_ed, ed_br))
        sc = asyncio.create_task(scan(sc_ed))


def main():
    from asciiapi import ASCIIDump
    app = App([ASCIIDump("/tmp/gotest")])
    asyncio.run(app.start())


if __name__ == '__main__':
    main()
