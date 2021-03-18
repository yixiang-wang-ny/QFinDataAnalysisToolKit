from abc import ABC, abstractmethod
from data.field import Field
import numpy as np


class Predictor(ABC):

    @abstractmethod
    def train(self, features: [Field], responses: [Field], **kwargs):
        pass

    @abstractmethod
    def predict(self, features: [Field]) -> np.array:
        pass

    @abstractmethod
    def get_name(self):
        pass
