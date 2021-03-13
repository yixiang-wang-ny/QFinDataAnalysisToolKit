from abc import ABC, abstractmethod

from typing import List

from data.feature import FeatureTemplate


class MissingValueHandler(ABC):

    @abstractmethod
    def train(self, features: List[FeatureTemplate]):
        pass

    @abstractmethod
    def apply(self, features: List[FeatureTemplate]):
        pass


class FillWithMedian(MissingValueHandler):

    def train(self, features: List[FeatureTemplate]):
        pass

    def apply(self, features: List[FeatureTemplate]):
        pass