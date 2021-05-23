from __future__ import annotations

from pathlib import Path
from typing import Type

import tensorflow as tf

from xidiff.equation import XiDiffEquation


# pylint: disable=too-many-ancestors
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

    def save(
        self: XiDiffModel,
        dir_path: Path
    ) -> None:
        model_path = dir_path / "model.tf"
        super().save(model_path)

        self.equation.save(dir_path)

    @classmethod
    def load(
        cls: Type[XiDiffModel],
        dir_path: Path,
    ) -> XiDiffModel:
        equation = XiDiffEquation.load(dir_path)

        model_path = dir_path / "model.tf"

        model = tf.keras.models.load_model(model_path)
        model.equation = equation

        return model
