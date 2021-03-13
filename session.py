import pandas as pd
from data.data_layer import Data


class QFinDASession(object):

    def __init__(self):

        self.data_object_map = {}

    def add_data_from_csv(self, csv_file_name):

        data_object = Data()
        data_object.add_from_csv(csv_file_name)
        self.data_object_map[data_object.data_id] = data_object





