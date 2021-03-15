from pipeline import QFinPipe
from typing import List
from data.feature import FeatureTemplate


class PCA(QFinPipe):

    def __init__(self, *args, **kwargs):

        super(PCA, self).__init__(*args, **kwargs)

    def train(self, features: List[FeatureTemplate]):

        for feature in features:
            pass

        return self.apply(features)

    def apply(self, features: List[FeatureTemplate]):

        for feature in features:
            pass

        return features