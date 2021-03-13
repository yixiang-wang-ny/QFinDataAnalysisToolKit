import datetime as dt
import pandas as pd
from data.feature import FeatureOnPandasDataFrame
from collections import OrderedDict


class Data(object):

    def __init__(self, data_id=None):

        if data_id is None:
            self.data_id = int(dt.datetime.now().strftime('%Y%m%d%H%M%S'))

        self.feature_map = OrderedDict()

    def add_from_data_frame(self, df, exclude_fields):

        for col in df:

            if col in exclude_fields:
                continue

            feature = FeatureOnPandasDataFrame(col, df)

            self.feature_map[feature.name] = feature

    def get_data_array(self, field):

        return self.feature_map[field].get_data()

    def get_data_frame(self, field):

        return pd.DataFrame({field: self.feature_map[field].get_data()})

    def get_columns(self):

        return self.feature_map.keys()
