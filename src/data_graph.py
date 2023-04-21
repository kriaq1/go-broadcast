import asyncio


class DataGraph:
    class Exception(Exception):
        def __init__(self, message):
            Exception.__init__(self, message)

    def __init__(self):
        self.components: dict[type, object] = {}

        self.producers: dict[(type, type), object] = {}
        self.consumers: dict[(type, type), object] = {}

        self.queues: dict[(type, type), asyncio.Queue] = {}

    def add_component(self, comp):
        if type(comp) in self.components:
            raise DataGraph.Exception("Component already exists")
        self.components[type(comp)] = comp
        return self

    def link_producer(self, src: type, dst: type):
        def linker(f):
            async def producer(comp, q):
                while True:
                    await q.put(f(comp))

            if (src, dst) in self.producers:
                raise DataGraph.Exception("producer already exists")
            self.producers[(src, dst)] = producer
            return f

        return linker

    def link_consumer(self, src: type, dst: type):
        def linker(f):
            async def consumer(comp, q):
                while True:
                    f(comp, await q.get())
                    q.task_done()

            if (src, dst) in self.consumers:
                raise DataGraph.Exception("consumer already exists")
            self.consumers[(src, dst)] = consumer
            return f

        return linker

    def run(self):
        ioloop = asyncio.get_event_loop()
        if {k for k in self.producers} != {k for k in self.consumers}:
            # TODO: list those components
            raise DataGraph.Exception("some components are not linked both ways")
        self.queues = {k: asyncio.Queue() for k in self.producers.keys()}
        producers_tasks = [
            ioloop.create_task(self.producers[k](self.components[k[0]], self.queues[k]))
            for k in self.producers.keys()
        ]
        consumer_tasks = [
            ioloop.create_task(self.consumers[k](self.components[k[1]], self.queues[k]))
            for k in self.producers.keys()
        ]
        # waits = asyncio.wait(producers_tasks + consumer_tasks)
        ioloop.run_forever()
