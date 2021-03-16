from model.predictor import Predictor
from data.field import Field
from pygam import LinearGAM, LogisticGAM, s, f, te
import pandas as pd
import numpy as np

TYPE_LINEAR = 'Linear'
TYPE_LOGISTIC = 'Logistic'


class GAM(Predictor):

    def __init__(self, model_type='Linear', num_splines=20, lam=None):

        self.lam = lam
        self.model_type = model_type
        self.num_splines = num_splines
        self.model = None

    @classmethod
    def get_gam_model(cls, features: [Field], model_type=TYPE_LINEAR):

        model_spec = f(0) if features[0].is_factor() else s(0)

        for i in range(1, len(features)):
            model_spec += f(0) if features[i].is_factor() else s(0)

        if model_type == TYPE_LINEAR:
            return LinearGAM(model_spec)

        if model_type == TYPE_LOGISTIC:
            return LogisticGAM(model_spec)

    def train(self, features: [Field], responses: [Field], **kwargs):

        self.model = self.get_gam_model(features, model_type=kwargs.get('model_type', TYPE_LINEAR))

        train_variates = pd.concat([x.data for x in features], axis=1)
        train_out = pd.concat([x.data for x in responses], axis=1)

        if self.lam is not None:
            lams = [np.array([self.lam])] * (train_variates.shape[1])
            self.model.gridsearch(train_variates.values, train_out.values, lam=lams)
        else:
            self.model.gridsearch(train_variates.values, train_out.values)

    def predict(self, features: [Field]):

        return self.model.predict(pd.concat([x.data for x in features], axis=1).values)
