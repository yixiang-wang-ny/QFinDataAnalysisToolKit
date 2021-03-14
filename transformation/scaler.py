from pipeline import QFinPipe
from typing import List
from data.feature import FeatureTemplate
from abc import abstractmethod
from collections import namedtuple

FeatureStats = namedtuple("FeatureStats", ("expectation", "deviation"))


class ExpectationDeviationScaler(QFinPipe):

    def __init__(self, *args, **kwargs):

        self.parameters_map: {str: FeatureStats} = {}

        super(ExpectationDeviationScaler, self).__init__(*args, **kwargs)

    @abstractmethod
    def stats(self, feature: FeatureTemplate) -> FeatureStats:
        pass

    def train(self, features: List[FeatureTemplate]):

        for feature in features:
            self.parameters_map[feature.name] = self.stats(feature)

        return self.apply(features)

    def apply(self, features: List[FeatureTemplate]):

        for feature in features:

            stats_obj = self.parameters_map[feature.name]

            feature.data = (feature.data - stats_obj.expectation) / stats_obj.deviation

        return features

