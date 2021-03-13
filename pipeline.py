from abc import ABC
from collections.abc import Iterable


class QFinPipeTemplate(ABC):

    pass


class QFinPipeLineRunner(object):

    def __init__(self):

        self.pipe_items = []

    def add(self, pipe_layer: [QFinPipeTemplate] or QFinPipeTemplate):

        self.pipe_items.append(pipe_layer)


class QFinPipe(QFinPipeTemplate):

    def __init__(self):

        self.down_stream_pipelines: [QFinPipeTemplate] = []

    def input(self):

        pass

    def append(self, downstream_q_fin_pipes: [QFinPipeTemplate] or QFinPipeTemplate) -> QFinPipeTemplate:
        if isinstance(downstream_q_fin_pipes, Iterable):
            self.down_stream_pipelines.extend(downstream_q_fin_pipes)
        else:
            self.down_stream_pipelines.append(downstream_q_fin_pipes)

        return self

    def train(self):

        pass

    def apply(self):

        pass


class TestQFinPipe1(QFinPipe):

    pass


class TestQFinPipe2(QFinPipe):
    pass


class TestQFinPipe3(QFinPipe):
    pass


class TestQFinPipe4(QFinPipe):
    pass


class TestQFinPipe5(QFinPipe):
    pass

