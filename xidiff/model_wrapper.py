from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence, Type, Union

import numpy as np
import tensorflow as tf

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
        return result

    def save(
        self: XiDiffModelWrapper,
        dir_path: Path,
    ) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)

        self._model.save(dir_path)

    @classmethod
    def load(
        cls: Type[XiDiffModelWrapper],
        dir_path: Path,
    ) -> XiDiffModelWrapper:
        model = XiDiffModel.load(dir_path)
        return XiDiffModelWrapper(model)
