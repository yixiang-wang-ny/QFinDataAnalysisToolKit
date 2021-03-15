from abc import ABC


class ValidationGenerator(ABC):

    pass


class RollingWindowGenerator(ValidationGenerator):

    def __init__(self, ts_ids, securities_ids, target_columns, features):

        self.ts_ids = ts_ids
        self.securities_ids = securities_ids
        self.target_columns = target_columns
        self.features = features

    def gen(self, train_window_size=40, test_window_size=5, step=1):

        return