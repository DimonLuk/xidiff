# pylint: disable=unused-argument
#pylint: disable=invalid-name
from typing import Final, List, Tuple

import numpy as np
import tensorflow as tf


def function_r(
    xs: tf.Tensor,
    fs: tf.Tensor,
    dfs: tf.Tensor,
    d2fs: tf.Tensor,
) -> List[tf.Tensor]:
    return [
        2*fs[:, 1]*d2fs[:, 1, 0, 1]*d2fs[:, 0, 0, 0] + fs[:, 0] *
        (tf.square(d2fs[:, 1, 0, 1]) + tf.square(d2fs[:, 0, 0, 1]))
    ]


def function_i(
    xs: tf.Tensor,
    fs: tf.Tensor,
    dfs: tf.Tensor,
    d2fs: tf.Tensor,
) -> List[tf.Tensor]:
    return [
        2*fs[:, 0]*d2fs[:, 1, 0, 1]*d2fs[:, 0, 0, 1] + fs[:, 1] *
        (tf.square(d2fs[:, 1, 0, 1]) + tf.square(d2fs[:, 0, 0, 1]))
    ]


def boundary_function(tf_model: tf.keras.models.Model) -> tf.Tensor:
    bc_input = tf.ones((1, 2), dtype="float64")

    with tf.GradientTape() as gradient:
        gradient.watch(bc_input)
        tmp_result = tf_model(bc_input, training=True)

    derivative_result = gradient.batch_jacobian(tmp_result, bc_input)
    derivative_result = tf.square(derivative_result)
    derivative_result = tf.reduce_sum(derivative_result)

    bc_result = tf.square(tmp_result)
    bc_result = tf.reduce_sum(bc_result)

    return derivative_result + bc_result


VARIABLE_RANGES: Final[List[Tuple[float, float]]] = [(-2, -1), (-3, -2)]

NUMBER_OF_FUNCTIONS: Final[int] = 1

ORDER_OF_SYSTEM: Final[int] = 1

EVALUATION_POINT_NUMPY: Final[List[int]] = [-1, -2]

EVALUATION_RANGE_NUMPY: Final[np.array] = np.array([[-1, -2], [-2, -3]])

EVALUATION_POINT_TENSORFLOW: Final[tf.Tensor] = tf.convert_to_tensor(
    EVALUATION_RANGE_NUMPY, dtype="float64")

EVALUATION_RANGE_TENSORFLOW: Final[tf.Tensor] = tf.convert_to_tensor(
    EVALUATION_RANGE_NUMPY, dtype="float64")
