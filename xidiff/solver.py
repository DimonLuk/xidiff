from __future__ import annotations

from typing import Optional, Type

import numpy as np
import tensorflow as tf

from xidiff.equation import XiDiffEquation
from xidiff.model import XiDiffModel
from xidiff.model_wrapper import XiDiffModelWrapper
from xidiff.train_step import get_train_step


class XiDiffSolver:
    # pylint: disable=too-few-public-methods
    def __init__(
        #pylint: disable=too-many-arguments
        self: XiDiffSolver,
        equation: XiDiffEquation,
        epochs: int = 50_000,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        losses_metric: Optional[tf.keras.metrics.Metric] = None,
        model_class: Type[XiDiffModel] = XiDiffModel,
        target_loss_exponent: int = -6,
    ) -> None:
        self.equation = equation
        self.epochs = epochs
        if optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=0.04, amsgrad=True)
        else:
            self.optimizer = optimizer

        if losses_metric is None:
            self.losses_metric = tf.keras.metrics.Mean(name="losses_metric")
        else:
            self.losses_metric = losses_metric

        self.model_class = model_class
        self.target_loss_exponent = target_loss_exponent

    def approximate(self: XiDiffSolver) -> XiDiffModelWrapper:
        values = self.equation.variables.get_tensor_values()
        model = self.model_class(self.equation)
        train_step = get_train_step()
        success_counter = 0

        for epoch in range(self.epochs):
            self.losses_metric.reset_states()
            train_step(values, model, self.optimizer, self.losses_metric)

            losses_result = self.losses_metric.result().numpy()

            if np.floor(np.log10(np.abs(losses_result))
                        ) <= self.target_loss_exponent:
                success_counter += 1

            if success_counter == 1000:
                break
            print(
                f"Epoch: {epoch + 1}\n"
                f"Loss: {losses_result}"
            )
            print("-" * 50)
        return XiDiffModelWrapper(model)
