# pylint: disable=invalid-name
from typing import Final, List, Tuple

import numpy as np
import tensorflow as tf


def function_r(
    xs: tf.Tensor,
    fs: tf.Tensor,
    dfs: tf.Tensor,
) -> List[tf.Tensor]:
    return [
        fs[:, 0] + dfs[:, 0, 0]*xs[:, 0],
        fs[:, 0] + dfs[:, 0, 1]*xs[:, 1],
    ]


def function_i(
    xs: tf.Tensor,
    fs: tf.Tensor,
    dfs: tf.Tensor,
) -> List[tf.Tensor]:
    return [
        fs[:, 1] + dfs[:, 1, 0]*xs[:, 0],
        fs[:, 1] + dfs[:, 1, 1]*xs[:, 1],
    ]


def boundary_function(tf_model: tf.keras.models.Model) -> tf.Tensor:
    bc_input = tf.ones((1, 2), dtype="float64")
    expected_outputs = [tf.ones((1, 1), dtype="float64")
                        * 3, tf.zeros((1, 1), dtype="float64")]
    # line below is correct but pylint is not satisfied
    # pylint: disable=no-value-for-parameter,unexpected-keyword-arg
    expected_outputs = tf.concat(expected_outputs, axis=1)

    bc_result = tf_model(bc_input, training=True)
    bc_result = tf.square(bc_result - expected_outputs)

    boundary_loss = tf.reduce_sum(bc_result)
    return boundary_loss


VARIABLE_RANGES: Final[List[Tuple[float, float]]] = [(1, 2), (1, 2)]

NUMBER_OF_FUNCTIONS: Final[int] = 1

ORDER_OF_SYSTEM: Final[int] = 2

EVALUATION_POINT_NUMPY: Final[List[int]] = [1, 1]

EVALUATION_RANGE_NUMPY: Final[np.array] = np.array([[1, 1], [1.5, 1.5]])

EVALUATION_POINT_TENSORFLOW: Final[tf.Tensor] = tf.convert_to_tensor(
    [EVALUATION_POINT_NUMPY], dtype="float64")

EVALUATION_RANGE_TENSORFLOW: Final[tf.Tensor] = tf.convert_to_tensor(
    EVALUATION_RANGE_NUMPY, dtype="float64")
