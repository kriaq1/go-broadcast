import unittest
from src.data_graph import DataGraph
from io import StringIO
import sys
import asyncio

import pdb


class TestDataGraph(unittest.TestCase):
    def test_two(self):
        # breakpoint()

        class A:
            def __init__(self):
                self.x = 0

        class B:
            def __init__(self):
                self.log = []

        graph = DataGraph()
        a = A()
        b = B()
        graph.add_component(a).add_component(b)

        increments = []
        consumes   = []

        output = StringIO()

        @graph.link_producer(A, B)
        async def increment(ax: A):
            if ax.x >= 100:
                return None
            ax.x += 1
            print("prod:", ax.x, file=output)
            increments.append(ax.x)
            return ax.x

        @graph.link_consumer(A, B)
        def echo(bx: B, x):
            bx.log.append(x)
            print("cons:", *bx.log, file=output)
            consumes.append(x)

        graph.run()
        self.assertTrue(increments == consumes)


if __name__ == '__main__':
    unittest.main()
