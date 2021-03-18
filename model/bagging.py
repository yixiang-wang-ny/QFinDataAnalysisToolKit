from abc import ABC, abstractmethod
from data.field import Field

from model.predictor import Predictor
from typing import ClassVar, List, Dict
import numpy as np
import datetime as dt


class Bagging(Predictor):

    model_cls: ClassVar[Predictor] = None
    model_args = None
    model_kwargs = None
    model_instance_map: Dict[object: Predictor] = None

    def __init__(self):

        raise Exception("Please use factory method")

    @classmethod
    def wrap(cls, model_class: ClassVar[Predictor], *model_args, **model_kwargs):

        obj = cls.__new__(cls)
        obj.model_cls = model_class
        obj.model_args = model_args
        obj.model_kwargs = model_kwargs
        obj.model_instances = []

        return obj

    @abstractmethod
    def aggregate(self, features: [Field]) -> np.array:

        return

    def train(self, features: [Field], responses: [Field], **kwargs):

        model_id = kwargs.get('model_id', int(dt.datetime.now().strftime('%Y%m%d%H%M%S')))

        model_instance = self.model_cls(*self.model_args, **self.model_kwargs)
        self.model_instance_map[model_id] = model_instance.train(features, responses, **kwargs)

    def predict(self, features: [Field]) -> np.array:
        return self.aggregate(features)

    def get_name(self):

        return "ABagOf{}".format(self.model_cls(*self.model_args, **self.model_kwargs).get_name());





