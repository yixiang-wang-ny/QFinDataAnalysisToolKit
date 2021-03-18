from abc import ABC, abstractmethod
import numpy as np


class PerformanceMeasure(ABC):

    @abstractmethod
    def score(self, predicted: np.array, true_response: np.array) -> float:
        pass

    @abstractmethod
    def get_name(self):
        return


class DirectionalAccuracy(PerformanceMeasure):

    def get_name(self):
        return "DirectionalAccuracy"

    def score(self, predicted: np.array, true_response: np.array):

        predict_zero_one_out = np.array([0 if x < 0 else 1 for x in predicted])
        true_zero_one_out = np.array([0 if x < 0 else 1 for x in true_response])

        return sum(predict_zero_one_out == true_zero_one_out) / float(len(predict_zero_one_out))


