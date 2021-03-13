from abc import ABC, abstractmethod
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import numpy as np
import pandas as pd


class FeatureTemplate(ABC):

    name = None

    @abstractmethod
    def get_data(self) -> np.array:
        return

    @abstractmethod
    def is_numeric(self):
        return

    @abstractmethod
    def is_factor(self):
        return

    @abstractmethod
    def has_missing_value(self):
        return


class FeatureOnPandasDataFrame(FeatureTemplate):

    def __init__(self, name: str, underlying_data_frame: pd.DataFrame):

        self.name = name
        self.underlying_data_frame = underlying_data_frame
        self._has_missing = self.underlying_data_frame[self.name].isnull().any()
        self._is_numeric = is_numeric_dtype(self.underlying_data_frame[self.name])
        self._is_factor = not self._is_numeric

    def get_data(self):
        return self.underlying_data_frame[self.name].values

    def is_numeric(self):
        return self._is_numeric

    def is_factor(self):
        return self._is_factor

    def has_missing_value(self):
        return self._has_missing
