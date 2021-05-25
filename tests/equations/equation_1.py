# pylint: disable=invalid-name
# pylint: disable=unused-argument
from typing import Final, List, Tuple

import numpy as np
import tensorflow as tf


def function_r(
    xs: tf.Tensor,
    fs: tf.Tensor,
    dfs: tf.Tensor,
) -> List[tf.Tensor]:
    return [
        dfs[:, 0, 0] - xs[:, 0]
    ]


def function_i(
    xs: tf.Tensor,
    fs: tf.Tensor,
    dfs: tf.Tensor,
) -> List[tf.Tensor]:
    return [
        dfs[:, 1, 0] - xs[:, 0]
    ]


def boundary_function(tf_model: tf.keras.models.Model) -> tf.Tensor:
    bc_input = tf.ones((1, 1), dtype="float64")

    bc_result = tf_model(bc_input, training=True)
    bc_result = tf.square(bc_result)

    return tf.reduce_sum(bc_result)


VARIABLE_RANGES: Final[List[Tuple[float, float]]] = [(-2, 2)]

NUMBER_OF_FUNCTIONS: Final[int] = 1

ORDER_OF_SYSTEM: Final[int] = 1

EVALUATION_POINT_NUMPY: Final[List[int]] = [1]

EVALUATION_RANGE_NUMPY: Final[np.array] = np.array([[1], [2]])

EVALUATION_POINT_TENSORFLOW: Final[tf.Tensor] = tf.convert_to_tensor(
    EVALUATION_RANGE_NUMPY, dtype="float64")

EVALUATION_RANGE_TENSORFLOW: Final[tf.Tensor] = tf.convert_to_tensor(
    EVALUATION_RANGE_NUMPY, dtype="float64")
