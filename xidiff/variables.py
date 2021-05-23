from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf


class XiDiffVariables:
    def __init__(
        self: XiDiffVariables,
        variable_ranges: Sequence[Tuple[float, float]],
        number_of_points: Optional[int] = None,
    ) -> None:
        self._variable_ranges = variable_ranges
        if number_of_points is None:
            self._number_of_points = self._generate_default_number_of_points()
        else:
            self._number_of_points = number_of_points

    def _generate_default_number_of_points(
        self: XiDiffVariables,
    ) -> int:
        max_diff = -1
        for range_ in self._variable_ranges:
            diff = np.abs(range_[0] - range_[1])
            if diff > max_diff:
                max_diff = diff

        exponent = np.floor(np.log10(max_diff))
        return int(np.power(10, exponent + 2))

    def get_tensor_values(self: XiDiffVariables) -> tf.Tensor:
        values = self.get_numpy_values()
        values = tf.convert_to_tensor(values, dtype="float64")
        return values

    def get_numpy_values(self: XiDiffVariables) -> np.array:
        lower_bounds = []
        upper_bounds = []
        for range_ in self._variable_ranges:
            lower_bounds.append(range_[0])
            upper_bounds.append(range_[1])
        values = np.linspace(lower_bounds, upper_bounds,
                             num=self._number_of_points)
        return values

    def __len__(self: XiDiffVariables) -> int:
        return len(self._variable_ranges)
