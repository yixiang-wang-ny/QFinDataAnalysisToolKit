import datetime as dt
import pandas as pd
from data.field import Field
from collections import OrderedDict
import data.generator as generator
from typing import Iterable
from copy import deepcopy


class Data(object):

    def __init__(self, data_id=None):

        if data_id is None:
            self.data_id = int(dt.datetime.now().strftime('%Y%m%d%H%M%S'))

        self.securities_id_columns = []
        self.ts_id_columns = []
        self.target_columns = []
        self.field_map = OrderedDict()

    def split_by_ts_id(self, ts_id_cut_off):

        columns = [self.field_map[f].data.name for f in self.ts_id_columns]
        ts_id_df = pd.concat(
            [self.field_map[f].data.reset_index(drop=True) for f in self.ts_id_columns], axis=1
        ).reset_index()
        ts_id_df.set_index(columns, inplace=True)

        if isinstance(ts_id_cut_off, Iterable):
            ts_id_cut_off = tuple(ts_id_cut_off)

        row_cut_off = ts_id_df.loc[ts_id_cut_off:].iloc[0]['index']
        return self.split_by_row(row_cut_off)

    def split_by_row(self, row_cut_off):

        row_end = len(list(self.field_map.values())[0].data)

        split = Data.__new__(Data)
        split.data_id = "{}-{}".format(self.data_id, 'split-{}-to-{}'.format(row_cut_off, row_end))
        split.securities_id_columns = deepcopy(self.securities_id_columns)
        split.ts_id_columns = deepcopy(self.ts_id_columns)
        split.target_columns = deepcopy(self.target_columns)
        split.field_map = OrderedDict()

        for k, v in self.field_map.items():
            split.field_map[k] = v.slice(row_cut_off, row_end)
            self.field_map[k] = v.slice(0, row_cut_off-1)

        return split

    def add_from_data_frame(self, df, exclude_fields, factor_fields=()):

        factor_fields = set(factor_fields)

        for col in df:

            if col in exclude_fields:
                continue

            field = Field.from_data_frame(col, df)
            if field.name in factor_fields:
                field.reset_meta(isFactor=True)

            self.field_map[field.name] = field

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

        return self.field_map[field].get_data()

    def get_data_frame(self, field):

        return pd.DataFrame({field: self.field_map[field].get_data()})

    def get_all_features(self) -> [Field]:
        return [
            v for k, v in self.field_map.items()
            if k not in self.securities_id_columns and k not in self.ts_id_columns and k not in self.target_columns
        ]

    def get_all_features_df(self) -> pd.DataFrame:

        return pd.concat([f.data for f in self.get_all_features()], axis=1)

    def get_ts_ids(self) -> [Field]:

        return [v for k, v in self.field_map.items() if k in self.ts_id_columns]

    def get_ts_ids_df(self) -> pd.DataFrame:

        return pd.concat([f.data for f in self.get_ts_ids()], axis=1)

    def get_securities_ids(self) -> [Field]:

        return [v for k, v in self.field_map.items() if k in self.securities_id_columns]

    def get_securities_ids_df(self) -> pd.DataFrame:

        return pd.concat([f.data for f in self.get_securities_ids()], axis=1)

    def get_target_fields(self) -> [Field]:

        return [v for k, v in self.field_map.items() if k in self.target_columns]

    def get_target_fields_df(self) -> pd.DataFrame:

        return pd.concat([f.data for f in self.get_target_fields()], axis=1)

    def get_field_names(self):

        return self.field_map.keys()

    def get_rolling_window_generator(self, train_window_size=40, test_window_size=5, step=1):

        gen_obj = generator.RollingWindowGenerator(
            self.get_ts_ids(),
            self.get_securities_ids(),
            self.get_target_fields(),
            self.get_all_features()
        )

        return gen_obj.gen(train_window_size=train_window_size, test_window_size=test_window_size, step=step)
