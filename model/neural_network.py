from model.predictor import Predictor
from data.field import Field
import pandas as pd
import numpy as np
from tensorflow import keras
from typing import List


class VanillaMultiLayerPerceptron(Predictor):

    def __init__(self, hidden_layer_num_units: List[int], output_activation: str = 'linear',
                 hidden_layer_activation: str = 'relu',
                 use_bias: bool = True, loss: str = 'categorical_crossentropy'):
        self.model = None
        self.hidden_layer_num_units = hidden_layer_num_units
        self.output_activation = output_activation
        self.hidden_layer_activation = hidden_layer_activation
        self.use_bias = use_bias
        self.loss = loss

    def get_name(self):
        return "MultiLayerPerceptron"

    def get_model(self, input_size):

        inputs = keras.Input(shape=input_size)

        x = keras.layers.Dense(
            self.hidden_layer_num_units[0], activation=self.hidden_layer_activation, use_bias=self.use_bias
        )(inputs)

        for i in range(len(self.hidden_layer_num_units)):
            x = keras.layers.Dense(
                self.hidden_layer_num_units[i], activation=self.hidden_layer_activation, use_bias=self.use_bias
            )(x)

        outputs = keras.layers.Dense(1, activation=self.output_activation, use_bias=self.use_bias)(x)

        model = keras.Model(inputs, outputs)

        model.compile(loss=self.loss)

        return model

    def train(self, features: [Field], responses: [Field], **kwargs):

        self.model = self.get_model(len(features))

        train_variates = pd.concat([x.data for x in features], axis=1)
        train_out = pd.concat([x.data for x in responses], axis=1)

        epochs = kwargs.get('epochs', 10)

        self.model.fit(train_variates.to_numpy(), train_out.to_numpy(), epochs=epochs)

    def predict(self, features: [Field]):

        return self.model.predict(Field.to_array(features)).flatten()

