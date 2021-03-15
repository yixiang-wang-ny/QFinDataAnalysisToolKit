from abc import ABC
import pandas as pd


class ValidationGenerator(ABC):

    pass


class RollingWindowGenerator(ValidationGenerator):

    def __init__(self, ts_ids, securities_ids, target_columns, features):

        self.ts_ids = ts_ids
        self.securities_ids = securities_ids
        self.target_columns = target_columns
        self.features = features

    def gen(self, train_window_size=40, test_window_size=5, step=1):

        ts_df = pd.concat([s.data for s in self.ts_ids], axis=1).reset_index(drop=True)
        ts_columns = ts_df.columns.tolist()
        ts_df_idx_map = ts_df.reset_index().set_index(ts_columns)

        distinct_ts_df = ts_df.drop_duplicates().set_index(ts_columns)
        num_ts = len(distinct_ts_df)

        for i in range(0, num_ts-train_window_size-test_window_size, step):

            train_start = i
            train_end = train_start + train_window_size - 1
            test_start = train_end + 1
            test_end = test_start + test_window_size - 1

            train_ids = distinct_ts_df.iloc[train_start: train_end].join(ts_df_idx_map)
            test_ts = distinct_ts_df.iloc[test_start: test_end].join(ts_df_idx_map)

            yield i