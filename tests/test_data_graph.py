import unittest
from src.data_graph import DataGraph
from io import StringIO
import sys
import asyncio

import pdb


class TestDataGraph(unittest.TestCase):
    class Producer:
        def __init__(self):
            self.x = 0

    class Consumer:
        def __init__(self):
            self.log = []

    def test_two(self):
        graph = DataGraph()
        a = self.Producer()
        b = self.Consumer()
        graph.add_component(a).add_component(b)

        increments = []
        consumes = []

        output = StringIO()

        @graph.link_producer(self.Producer, self.Consumer)
        def increment(ax: self.Producer):
            if ax.x >= 100:
                return None
            ax.x += 1
            print("prod:", ax.x, file=output)
            increments.append(ax.x)
            return ax.x

        @graph.link_consumer(self.Producer, self.Consumer)
        def echo(bx: self.Consumer, x):
            bx.log.append(x)
            print("cons:", *bx.log, file=output)
            consumes.append(x)

        graph.run()
        print(*increments)
        print(*consumes)
        self.assertTrue(increments == consumes)

    def test_exception(self):
        graph = DataGraph()
        a = self.Producer()
        b = self.Consumer()

        @graph.link_producer(type(a), type(b))
        def increment(prod: type(a)):
            if ax.x >= 100:
                # ax.x /= 0
                return None
            ax.x += 1
            return ax.x

        @graph.link_consumer(type(a), type(b))
        def recive(cons: type(b), x):
            pass

        graph.run()

        # try:
        #     graph.run()
        #     self.fail()
        # except ZeroDivisionError:
        #     print("/0")
        # except Exception as exp:
        #     print("Exception passed:", exp.args)
        #     pass

    def test_yield(self):
        graph = DataGraph()
        prod = self.Producer()
        b = self.Consumer()


if __name__ == '__main__':
    unittest.main()
