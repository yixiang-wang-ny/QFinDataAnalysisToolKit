from abc import ABC, abstractmethod
from pandas.api.types import is_numeric_dtype
import numpy as np
import pandas as pd
import datetime as dt


class Field(object):

    name = None
    data: pd.Series = None
    _has_missing = None
    _is_numeric = None
    _is_factor = None

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __init__(self):
        pass
        # raise Exception("Please use one of the factory method")

    @classmethod
    def from_data_frame(cls, name, underlying_data_frame: pd.DataFrame):

        obj = Field()

        obj.name = name
        obj.data = underlying_data_frame[name].reset_index(drop=True)
        obj._has_missing = obj.data.isnull().any()
        obj._is_numeric = is_numeric_dtype(obj.data)
        obj._is_factor = not obj._is_numeric

        return obj

    def get_data(self) -> np.array:
        return self.data.values

    def is_numeric(self):
        return self._is_numeric

    def is_factor(self):
        return self._is_factor

    def has_missing_value(self):
        return self._has_missing
