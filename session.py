import pandas as pd
from data.data_layer import Data
from collections.abc import Iterable


class QFinDASession(object):

    def __init__(self):

        self.data_layer = None
        self.securities_ids = []
        self.ts_ids = []
        self.target_columns = []

    def add_data_from_data_frame(self, df, exclude_fields=()):

        data_object = Data()
        data_object.add_from_data_frame(df, exclude_fields)
        self.data_layer = data_object

    def set_time_series_id(self, ts_id_names):

        if isinstance(ts_id_names, Iterable):
            self.ts_ids.extend(ts_id_names)
        else:
            self.ts_ids.append(ts_id_names)

    def set_target_columns(self, target_columns):
        if isinstance(target_columns, Iterable):
            self.target_columns.extend(target_columns)
        else:
            self.target_columns.append(target_columns)

    @property
    def data(self):
        return self.data_layer





