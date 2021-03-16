from abc import ABC, abstractmethod
from collections.abc import Iterable
from data.field import Field


class QFinPipe(ABC):

    def __init__(self, input_features=None):

        self.input_features = input_features
        self.down_stream_pipelines: [QFinPipe] = []

    @abstractmethod
    def train(self, features: [Field]) -> [Field]:
        pass

    @abstractmethod
    def apply(self, features: [Field]) -> [Field]:
        pass

    def append(self, downstream_q_fin_pipes):

        downstream_q_fin_pipes: [QFinPipe] or QFinPipe

        if isinstance(downstream_q_fin_pipes, Iterable):
            self.down_stream_pipelines.extend(downstream_q_fin_pipes)
        else:
            self.down_stream_pipelines.append(downstream_q_fin_pipes)

        return self

    def train_pipe(self, features: [Field]):

        if self.input_features is None:
            features = self.train(features)
        else:
            features = self.train([x for x in features if x.name in self.input_features])

        if len(self.down_stream_pipelines) == 0:
            return features

        down_stream_res = []
        for down_stream_pipe in self.down_stream_pipelines:
            down_stream_res.extend(down_stream_pipe.train_pipe(features))

        return down_stream_res


class PipeSelect(QFinPipe):

    def train(self, features: [Field]) -> [Field]:
        return features

    def apply(self, features: [Field]) -> [Field]:
        return features


class QFinPipeLine(object):

    def __init__(self):

        self.pipe_items: [[QFinPipe]] = []

    def add(self, pipe_layer: [QFinPipe] or QFinPipe):

        if isinstance(pipe_layer, Iterable):
            self.pipe_items.append(pipe_layer)
        else:
            self.pipe_items.append([pipe_layer])

    def train(self, features: [Field]) -> [Field]:

        for item in self.pipe_items:

            item_res = []
            for pipe in item:
                item_res.extend(pipe.train_pipe(features))

            features = item_res

        return features

