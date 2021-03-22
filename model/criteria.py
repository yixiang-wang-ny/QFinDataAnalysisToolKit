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

        predict_zero_one_out = np.array([0 if x <= 0 else 1 for x in predicted])
        true_zero_one_out = np.array([0 if x <= 0 else 1 for x in true_response])

        return sum(predict_zero_one_out == true_zero_one_out) / float(len(predict_zero_one_out))


class BuySignalDirectionalAccuracy(PerformanceMeasure):

    def get_name(self):
        return "BuySignalDirectionalAccuracy"

    def score(self, predicted: np.array, true_response: np.array):

        predict_zero_one_out = np.array([0 if x <= 0 else 1 for x in predicted])
        true_zero_one_out = np.array([0 if x <= 0 else 1 for x in true_response])

        correct_guess = 0
        wrong_guess = 0

        for i in range(len(predict_zero_one_out)):

            pred_i = predict_zero_one_out[i]
            true_i = true_zero_one_out[i]

            if pred_i == 1:
                if true_i == 1:
                    correct_guess += 1
                else:
                    wrong_guess += 1

        return correct_guess / float(correct_guess+wrong_guess)

