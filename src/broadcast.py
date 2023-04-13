from board import Turn
from api import API

import asyncio


class Broadcast:
    __slots__ = ["apis"]

    def __init__(self, apis: list[API]):
        self.apis = apis

    def add_turn(self, turn: Turn):
        for api in self.apis:
            api.add(turn)

    def broadcast(self):
        for api in self.api:
            api.broadcast()
