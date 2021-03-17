from pandas.api.types import is_numeric_dtype
import numpy as np
import pandas as pd
from collections import namedtuple
from copy import deepcopy

FieldMeta = namedtuple("FieldMeta", ("hasMissing", "isNumeric", "isFactor"))


class Field(object):

    name = None
    data: pd.Series = None
    meta: FieldMeta = None

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __init__(self):
        raise RuntimeError("Please use one of the factory methods")

    @classmethod
    def from_data_frame(cls, name, underlying_data_frame: pd.DataFrame):

        obj = cls.__new__(cls)

        obj.name = name
        obj.data = underlying_data_frame[name].reset_index(drop=True)

        obj.meta = FieldMeta(
            obj.data.isnull().any(),
            is_numeric_dtype(obj.data),
            not is_numeric_dtype(obj.data)
        )

        return obj

    def get_data(self) -> np.array:
        return self.data.values

    def is_numeric(self):
        return self.meta.isNumeric

    def is_factor(self):
        return self.meta.isFactor

    def has_missing_value(self):
        return self.meta.hasMissing

    def slice(self, start, end):

        obj = self.__new__(Field)

        obj.name = self.name
        obj.data = self.data.iloc[start: end]
        obj.meta = deepcopy(self.meta)

        return obj

