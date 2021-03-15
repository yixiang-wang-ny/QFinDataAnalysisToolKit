from pipeline import QFinPipe
from typing import List
from data.feature import FeatureTemplate, FeatureSeries
import pandas as pd
import sklearn.decomposition as decomposition


class PCA(QFinPipe):

    obj_counter = 0

    def __init__(self, *args, **kwargs):

        PCA.obj_counter += 1
        self.component_prefix = kwargs.get('component_prefix', 'PCA{}_'.format(PCA.obj_counter))
        self.pca_obj = None
        self.cum_variance_ratio = kwargs.get('cum_variance_ratio', 0.95)

        super(PCA, self).__init__(*args, **kwargs)

    def train(self, features: List[FeatureTemplate]):

        # todo: I am aiming to make it work at this moment, but should be made more efficient

        self.pca_obj = decomposition.PCA(self.cum_variance_ratio)
        self.pca_obj.fit(pd.concat([feature.data for feature in features], axis=1).values)

        return self.apply(features)

    def apply(self, features: List[FeatureTemplate]):

        # todo: same here, need to be optimized in terms of both memory and performance
        out_pcs = self.pca_obj.transform(pd.concat([feature.data for feature in features], axis=1).values)
        out_df = pd.DataFrame(out_pcs, columns=[self.component_prefix + str(i) for i in range(out_pcs.shape[1])])

        return [FeatureSeries(col, out_df) for col in out_df]



