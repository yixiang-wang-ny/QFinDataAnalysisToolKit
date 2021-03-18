from abc import ABC, abstractmethod
from data.field import Field
from typing import Iterable
from model.predictor import Predictor
from typing import ClassVar, List, Dict
import numpy as np
import datetime as dt


class Bagging(Predictor):

    model_cls: ClassVar[Predictor] = None
    model_args = None
    model_kwargs = None
    model_instance_map: Dict[object, Predictor] = None

    def __init__(self):

        raise Exception("Please use factory method")

    @classmethod
    def wrap(cls, model_class: ClassVar[Predictor], *model_args, **model_kwargs):

        obj = cls.__new__(cls)
        obj.model_cls = model_class
        obj.model_args = model_args
        obj.model_kwargs = model_kwargs
        obj.model_instance_map = {}

        return obj

    @abstractmethod
    def aggregate(self, features: [Field], models: List[Predictor]) -> np.array:

        return

    def train(self, features: [Field], responses: [Field], **kwargs):

        model_id = kwargs.get('model_id', int(dt.datetime.now().strftime('%Y%m%d%H%M%S')))

        model_instance = self.model_cls(*self.model_args, **self.model_kwargs)
        self.model_instance_map[model_id] = model_instance.train(features, responses, **kwargs)

    def predict(self, features: [Field]) -> np.array:
        return self.aggregate(features, list(self.model_instance_map.values()))

    def predict_with_exclusion(self, features: [Field], exclude: Iterable) -> np.array:
        return self.aggregate(features, [v for k, v in self.model_instance_map.items() if k not in exclude])

    def get_name(self):

        return "A Bag Of {}".format(self.model_cls(*self.model_args, **self.model_kwargs).get_name())


class BaggingByDirectionalVotes(Bagging):

    voting_cutoff = 0.5

    def aggregate(self, features: [Field], models: List[Predictor]) -> np.array:

        length = len(features[0].data)
        returns = np.zeros(length)
        votes = np.zeros(length)

        for _, model in self.model_instance_map:
            prediction = model.predict(features)
            returns += prediction
            votes += np.array([x > 0 for x in prediction])

        voted_zero_one_out_even = np.array([0 if x < int(len(models) * self.voting_cutoff) else 1 for x in votes])
        return voted_zero_one_out_even
