import datetime as dt
import pandas as pd
from data.feature import FeatureSeries
from collections import OrderedDict
import data.generator as generator


class Data(object):

    def __init__(self, data_id=None):

        if data_id is None:
            self.data_id = int(dt.datetime.now().strftime('%Y%m%d%H%M%S'))

        self.securities_id_columns = []
        self.ts_id_columns = []
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
            self.ts_id_columns.extend(ts_id_names)
        else:
            self.ts_id_columns.append(ts_id_names)

    def set_securities_ids(self, securities_id_names):

        if not isinstance(securities_id_names, str):
            self.securities_id_columns.extend(securities_id_names)
        else:
            self.securities_id_columns.append(securities_id_names)

    def set_target_fields(self, target_columns):

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
            if k not in self.securities_id_columns and k not in self.ts_id_columns and k not in self.target_columns
        ]

    def get_ts_ids(self):

        return [v for k, v in self.feature_map.items() if k in self.ts_id_columns]

    def get_securities_ids(self):

        return [v for k, v in self.feature_map.items() if k in self.securities_id_columns]

    def get_target_fields(self):

        return [v for k, v in self.feature_map.items() if k in self.target_columns]

    def get_field_names(self):

        return self.feature_map.keys()

    def get_rolling_window_generator(self, train_window_size=40, test_window_size=5, step=1):

        gen_obj = generator.RollingWindowGenerator(
            self.get_ts_ids(),
            self.get_securities_ids(),
            self.get_target_fields(),
            self.get_all_features()
        )

        yield gen_obj.gen(train_window_size=train_window_size, test_window_size=test_window_size, step=step)
