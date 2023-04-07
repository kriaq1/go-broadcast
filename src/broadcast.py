from board import Board
from api import API
from board_scanner import BoardScanner
from manual_editor import ManualEditor

import asyncio


class Broadcast:
    __slots__ = ["api", "app", "scanner", "editor"]

    def __init__(self, api: list[API], *scanner_args):
        self.scanner = BoardScanner(*scanner_args)
        self.api = api
        self.editor = ManualEditor()

    async def start(self):
        async def scan(scan_edit_queue):
            b = await self.scanner.get_board()
            while b:
                await scan_edit_queue.put(b)

        async def edit(scan_edit_queue, edit_broadcast_queue):
            b = await scan_edit_queue.get()
            self.editor.add(b)
            await edit_broadcast_queue.put(self.editor.get())
            scan_edit_queue.task_done()

        async def broadcast(edit_broadcast_queue):
            b = await edit_broadcast_queue.get()
            for api in self.api:
                try:
                    api.add(b)
                except Exception:
                    pass

        sc_ed = asyncio.Queue()
        ed_br = asyncio.Queue()

        async with asyncio.TaskGroup() as tg:
            sc = tg.create_task(scan(sc_ed))
            ed = tg.create_task(edit(sc_ed, ed_br))
            br = tg.create_task(broadcast(ed_br))


def main():
    from asciiapi import ASCIIDump
    broadcaster = Broadcast([ASCIIDump("/tmp/gotest")])
    asyncio.run(broadcaster.start())


if __name__ == '__main__':
    main()
