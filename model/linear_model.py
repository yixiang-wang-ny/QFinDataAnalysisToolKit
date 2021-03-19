from model.predictor import Predictor
from data.field import Field
import pandas as pd
from sklearn import linear_model


class LinearRegression(Predictor):

    def __init__(self):
        self.model = None

    def get_name(self):

        return "LinearRegression"

    def train(self, features: [Field], responses: [Field], **kwargs):

        self.model = linear_model.LinearRegression()

        train_variates = pd.concat([x.data for x in features], axis=1)
        train_out = pd.concat([x.data for x in responses], axis=1)

        self.model.fit(train_variates.to_numpy(), train_out.to_numpy())

    def predict(self, features: [Field]):

        return self.model.predict(pd.concat([x.data for x in features], axis=1).values)