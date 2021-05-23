from __future__ import annotations

from os import PathLike
from typing import TYPE_CHECKING, Literal, Sequence, Type, Union

import numpy as np
import tensorflow as tf

if TYPE_CHECKING:
    from xidiff.model import XiDiffModel

OutputDataStyle = Literal["numpy", "tensorflow"]


class XiDiffModelWrapper:
    def __init__(
        self: XiDiffModelWrapper,
        model: XiDiffModel,
    ) -> None:
        self._model = model

    def __call__(
        self: XiDiffModelWrapper,
        *values: Sequence[float],
        style: OutputDataStyle = "numpy",
    ) -> Union[np.array, tf.Tensor]:
        if np.array(values).shape == (len(self._model.equation.variables),):
            return self.evaluate_at(*values, style=style)
        else:
            return self.evaluate_range(values[0], style=style)

    def evaluate_at(
        self: XiDiffModelWrapper,
        *values: Sequence[float],
        style: OutputDataStyle = "numpy"
    ) -> Union[np.array, tf.Tensor]:
        values = tf.convert_to_tensor([values], dtype="float64")
        result = self._model(values)
        if style == "numpy":
            result = result.numpy()
            return result[0]
        elif style == "tensorflow":
            return result

    def evaluate_range(
        self: XiDiffModelWrapper,
        values: Union[np.array, tf.Tensor],
        style: OutputDataStyle = "numpy",
    ) -> Union[np.array, tf.Tensor]:
        if not isinstance(values, tf.Tensor):
            values = tf.convert_to_tensor(values, dtype="float64")

        result = self._model(values)
        if style == "numpy":
            return result.numpy()
        elif style == "tensorflow":
            return result

    def save(
        self: XiDiffModelWrapper,
        path: PathLike
    ) -> None:
        self._model.save(path)

    @classmethod
    def load(
        cls: Type[XiDiffModelWrapper],
        path: PathLike
    ) -> XiDiffModelWrapper:
        tf_model = tf.keras.models.load_model(path)
        return cls(tf_model)
