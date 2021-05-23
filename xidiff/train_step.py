# pylint: disable=invalid-name
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import tensorflow as tf

from xidiff.gradient_manager import XiDiffGradientManager

if TYPE_CHECKING:
    from xidiff.model import XiDiffModel


TrainStepFunction = Callable[
    [
        tf.Tensor,
        "XiDiffModel",
        tf.keras.optimizers.Optimizer,
        tf.keras.metrics.Metric,
    ],
    None
]


def get_train_step(
    performance: bool = True
) -> TrainStepFunction:
    def train_step(
        xs: tf.Tensor,
        model: XiDiffModel,
        optimizer: tf.keras.optimizers.Optimizer,
        losses_metric: tf.keras.metrics.Metric,
    ) -> None:
        with tf.GradientTape() as loss_gr:
            loss_gr.watch(model.trainable_variables)

            with XiDiffGradientManager(xs, model.equation) as xidiff_gradients:
                result = model(xs, training=True)

            # member is actually present
            # pylint: disable=no-member
            derivatives = xidiff_gradients.batch_jacobians(result)

            # calculate losses
            loss_real = tf.convert_to_tensor(
                model.equation.real_function(xs, result, *derivatives))
            loss_real = tf.reshape(loss_real, xs.shape)
            loss_real = tf.square(loss_real)

            loss_imaginary = tf.convert_to_tensor(
                model.equation.imaginary_function(xs, result, *derivatives))
            loss_imaginary = tf.reshape(loss_imaginary, xs.shape)
            loss_imaginary = tf.square(loss_imaginary)

            loss = tf.reduce_mean(loss_real + loss_imaginary)

            # calculate boundary conditions
            boundary_loss = model.equation.boundary_conditions_function(model)

            loss += boundary_loss

        losses_metric(loss)
        gradients = loss_gr.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if performance:
        return tf.function(train_step)
    return train_step
