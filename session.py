import pandas as pd
from data.data_layer import Data
from pipeline import QFinPipeLine
from model.predictor import Predictor
from model.criteria import PerformanceMeasure
from data.generator import ValidationGenerator, DataContainer
from typing import List, Optional, Generator
from collections import namedtuple
from typing import Dict, List
from itertools import chain


ModelValidationStats = namedtuple(
    "ModelValidationStats",
    ('name', 'train_ts_range', 'test_ts_range', 'train_securities_range', 'test_securities_range', 'trained_model',
     'score_map')
)


class QFinDASession(object):

    def __init__(self):

        self.data_layer = None
        self.feature_transformer = None
        self.models: List[Predictor] = []
        self.performance_measures: List[PerformanceMeasure] = []
        self.validation_data_generator: Optional[Generator[DataContainer]] = None
        self.model_train_history: Dict[str: List[ModelValidationStats]] = {}

    def add_data_from_csv(self, file_path, exclude_fields=(), factor_fields=()):

        df = pd.read_csv(file_path)
        self.add_data_from_data_frame(df, exclude_fields, factor_fields)

    def add_data_from_data_frame(self, df, exclude_fields=(), factor_fields=()):

        data_object = Data()
        data_object.add_from_data_frame(df, exclude_fields, factor_fields)
        self.data_layer = data_object

    @property
    def data(self):
        return self.data_layer

    def set_feature_transformer(self, transformer: [QFinPipeLine]):
        self.feature_transformer = transformer

    def run_feature_transformer(self):

        features = self.data.get_all_features()

        for f in features:
            self.data.field_map.pop(f.name)

        out_features = self.feature_transformer.train(features)

        for f in out_features:
            self.data.field_map[f.name] = f

    def add_model(self, model: Predictor):
        self.models.append(model)

    def add_model_performance_measure(self, measure: PerformanceMeasure):
        self.performance_measures.append(measure)

    def set_data_validation_generator(self, generator: ValidationGenerator):
        self.validation_data_generator = generator

    def kick_off_model_search(self):

        self.run_feature_transformer()
        for data in self.validation_data_generator:
            for model in self.models:
                model.train(data.train_in, data.train_out)

                model_name = model.get_name()
                if model_name not in self.model_train_history:
                    self.model_train_history[model_name] = []

                train_ts_range = (data.train_ts[0].data.min(), data.train_ts[0].data.max()) if data.train_ts else None
                test_ts_range = (data.test_ts[0].data.min(), data.test_ts[0].data.max()) if data.test_ts else None
                train_securities_range = (data.train_securities[0].data.min(), data.train_securities[0].data.max()) if data.train_securities else None
                test_securities_range = (data.test_securities[0].data.min(), data.test_ts[0].data.max()) if data.test_securities else None
                score_map = {
                    x.get_name(): x.score(
                        model.predict(data.test_in), pd.concat([x.data for x in data.test_out], axis=1).values
                    ) for x in self.performance_measures
                }

                self.model_train_history[model_name].append(
                    ModelValidationStats(
                        name=model_name, train_ts_range=train_ts_range, test_ts_range=test_ts_range,
                        train_securities_range=train_securities_range, test_securities_range=test_securities_range,
                        trained_model=model, score_map=score_map

                    )
                )
