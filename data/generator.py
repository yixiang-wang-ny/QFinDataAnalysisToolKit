from abc import ABC
import pandas as pd
from data.field import Field
from typing import Generator, Optional, List


class DataContainer(object):

    def __init__(
            self, train_in: List[Field], train_out: List[Field], test_in: List[Field], test_out: List[Field],
            train_ts: Optional[List[Field]], test_ts: Optional[List[Field]], train_securities: Optional[List[Field]],
            test_securities: Optional[List[Field]]
    ):

        self.train_in: List[Field] = train_in
        self.train_out: List[Field] = train_out
        self.test_in: List[Field] = test_in
        self.test_out: List[Field] = test_out
        self.train_ts: Optional[List[Field]] = train_ts
        self.test_ts: Optional[List[Field]] = test_ts
        self.train_securities: Optional[List[Field]] = train_securities
        self.test_securities: Optional[List[Field]] = test_securities


class ValidationGenerator(ABC):

    pass


class RollingWindowGenerator(ValidationGenerator):

    def __init__(self, ts_ids: [Field], securities_ids: [Field], target_columns: [Field], features: [Field]):

        self.ts_ids = ts_ids
        self.securities_ids = securities_ids
        self.target_columns = target_columns
        self.features = features

    def gen(self, train_window_size=40, test_window_size=5, step=1) -> Generator[DataContainer, None, None]:

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

            train_ids = distinct_ts_df.iloc[train_start: (train_end+1)].join(ts_df_idx_map)
            test_ids = distinct_ts_df.iloc[test_start: (test_end+1)].join(ts_df_idx_map)

            train_start_idx = train_ids['index'].min()
            train_end_idx = train_ids['index'].max()
            test_start_dix = test_ids['index'].min()
            test_end_idx = test_ids['index'].max()

            yield DataContainer(
                [f.slice(train_start_idx, train_end_idx+1) for f in self.features],
                [f.slice(train_start_idx, train_end_idx+1) for f in self.target_columns],
                [f.slice(test_start_dix, test_end_idx+1) for f in self.features],
                [f.slice(test_start_dix, test_end_idx+1) for f in self.target_columns],
                [f.slice(train_start_idx, train_end_idx+1) for f in self.ts_ids],
                [f.slice(test_start_dix, test_end_idx+1) for f in self.ts_ids],
                [f.slice(train_start_idx, train_end_idx + 1) for f in self.securities_ids],
                [f.slice(test_start_dix, test_end_idx + 1) for f in self.securities_ids]
            )


