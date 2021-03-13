import datetime as dt
import pandas as pd


class Data(object):

    def __init__(self, data_id=None):

        if data_id is None:
            self.data_id = int(dt.datetime.now().strftime('%Y%m%d%H%M%S'))

    def add_from_csv(self, csv_path):

        df = pd.read_csv(csv_path)
        return

