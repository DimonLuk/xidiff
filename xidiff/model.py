from __future__ import annotations

import tensorflow as tf

from xidiff.equation import XiDiffEquation


class XiDiffModel(tf.keras.Model):
    def __init__(
        self: XiDiffModel,
        equation: XiDiffEquation,
    ) -> None:
        super().__init__()
        self.equation = equation
        self.input_layer = tf.keras.layers.Dense(
            len(self.equation.variables), activation=None, dtype="float64")
        self.hidden_layers = [
            tf.keras.layers.Dense(10, activation="sigmoid", dtype="float64"),
            tf.keras.layers.Dense(10, activation="tanh", dtype="float64"),
        ]
        self.output_layer = tf.keras.layers.Dense(
            2 * self.equation.unknown_functions,
            activation=None,
            dtype="float64"
        )

    def call(self: XiDiffModel, x: tf.Tensor):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)
