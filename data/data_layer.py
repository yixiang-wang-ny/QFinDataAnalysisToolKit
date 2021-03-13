import datetime as dt
import pandas as pd
from data.feature import FeatureOnPandasDataFrame


class Data(object):

    def __init__(self, data_id=None):

        if data_id is None:
            self.data_id = int(dt.datetime.now().strftime('%Y%m%d%H%M%S'))

        self.feature_map = {}

    def add_from_csv(self, csv_path):

        df = pd.read_csv(csv_path)
        for col in df:
            feature = FeatureOnPandasDataFrame(col, df)

            self.feature_map[feature.name] = feature


