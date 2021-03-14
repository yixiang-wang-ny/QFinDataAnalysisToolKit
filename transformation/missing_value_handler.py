from pipeline import QFinPipe
from typing import List
from data.feature import FeatureTemplate
from abc import abstractmethod


class MissingValueHandler(QFinPipe):

    def __init__(self, *args, **kwargs):

        self.parameters_map: {str: float} = {}

        super(MissingValueHandler, self).__init__(*args, **kwargs)

    @abstractmethod
    def unconditional_expectation_measure(self, feature: FeatureTemplate):
        pass

    def train(self, features: List[FeatureTemplate]):

        for feature in features:
            self.parameters_map[feature.name] = self.unconditional_expectation_measure(feature)

        return self.apply(features)

    def apply(self, features: List[FeatureTemplate]):

        for feature in features:
            feature.data = feature.data.fillna(self.parameters_map[feature.name])

        return features


class FillWithMedian(MissingValueHandler):

    def unconditional_expectation_measure(self, feature: FeatureTemplate):
        return feature.data.median()


class FillWithMean(MissingValueHandler):

    def unconditional_expectation_measure(self, feature: FeatureTemplate):
        return feature.data.mean()

