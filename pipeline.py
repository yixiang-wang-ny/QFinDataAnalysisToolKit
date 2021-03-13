from abc import ABC, abstractmethod
from collections.abc import Iterable
from data.feature import FeatureTemplate


class QFinPipeTemplate(ABC):

    @abstractmethod
    def train(self, features: [FeatureTemplate]):
        pass


class QFinPipeLine(object):

    def __init__(self):

        self.pipe_items: [[QFinPipeTemplate]] = []

    def add(self, pipe_layer: [QFinPipeTemplate] or QFinPipeTemplate):

        if isinstance(pipe_layer, Iterable):
            self.pipe_items.append(pipe_layer)
        else:
            self.pipe_items.append([pipe_layer])

    def train(self, features: [FeatureTemplate]) -> [FeatureTemplate]:

        for item in self.pipe_items:

            item_res = []
            for pipe in item:
                item_res.extend(pipe.train(features))

            features = item_res

        return features


class QFinPipe(QFinPipeTemplate):

    def __init__(self):

        self.down_stream_pipelines: [QFinPipeTemplate] = []

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

