import pandas as pd
from data.data_layer import Data


class QFinDASession(object):

    def __init__(self):

        self.data_layer = None

    def add_data_from_data_frame(self, df, exclude_fields=()):

        data_object = Data()
        data_object.add_from_data_frame(df, exclude_fields)
        self.data_layer = data_object

    @property
    def data(self):
        return self.data_layer





