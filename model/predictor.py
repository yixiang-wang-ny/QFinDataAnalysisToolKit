from abc import ABC, abstractmethod
from data.field import Field


class Predictor(ABC):

    @abstractmethod
    def train(self, features: [Field], responses: [Field], **kwargs):
        pass

    @abstractmethod
    def predict(self, features: [Field]):
        pass

