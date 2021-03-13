import datetime as dt
import pandas as pd
from data.feature import FeatureSeries
from collections import OrderedDict
from collections.abc import Iterable


class Data(object):

    def __init__(self, data_id=None):

        if data_id is None:
            self.data_id = int(dt.datetime.now().strftime('%Y%m%d%H%M%S'))

        self.securities_ids = []
        self.ts_ids = []
        self.target_columns = []
        self.feature_map = OrderedDict()

    def add_from_data_frame(self, df, exclude_fields):

        for col in df:

            if col in exclude_fields:
                continue

            feature = FeatureSeries(col, df)

            self.feature_map[feature.name] = feature

    def set_time_series_id(self, ts_id_names):

        if not isinstance(ts_id_names, str):
            self.ts_ids.extend(ts_id_names)
        else:
            self.ts_ids.append(ts_id_names)

    def set_securities_ids(self, securities_id_names):

        if not isinstance(securities_id_names, str):
            self.securities_ids.extend(securities_id_names)
        else:
            self.securities_ids.append(securities_id_names)

    def set_target_columns(self, target_columns):

        if not isinstance(target_columns, str):
            self.target_columns.extend(target_columns)
        else:
            self.target_columns.append(target_columns)

    def get_data_array(self, field):

        return self.feature_map[field].get_data()

    def get_data_frame(self, field):

        return pd.DataFrame({field: self.feature_map[field].get_data()})

    def get_all_features(self):
        return [
            v for k, v in self.feature_map.items()
            if k not in self.securities_ids and k not in self.ts_ids and k not in self.target_columns
        ]

    def get_columns(self):

        return self.feature_map.keys()
