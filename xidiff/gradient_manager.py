# pylint: disable=invalid-name
from __future__ import annotations

import contextlib
from typing import List, Sequence

import tensorflow as tf

from xidiff.equation import XiDiffEquation


class XiDiffGradientManager(contextlib.ExitStack):
    def __init__(
        self: XiDiffGradientManager,
        xs: tf.Tensor,
        equation: XiDiffEquation,
    ) -> None:
        super().__init__()
        self.xs = xs
        self.equation = equation

        self._gradient_tapes = [tf.GradientTape() for _ in range(
            self.equation.highest_order_derivative)]

    def __enter__(self: XiDiffGradientManager) -> XiDiffGradientManager:
        stack = super().__enter__()

        for gradient_tape in self._gradient_tapes:
            tape = stack.enter_context(gradient_tape)
            tape.watch(self.xs)

        return stack

    def batch_jacobians(
        self: XiDiffGradientManager,
        result: tf.Tensor
    ) -> Sequence[tf.Tensor]:
        gradient_tapes = self._gradient_tapes[::-1]
        derivatives: List[tf.Tensor] = []
        for tape in gradient_tapes:
            if not derivatives:
                derivatives.append(tape.batch_jacobian(result, self.xs))
            else:
                derivatives.append(tape.batch_jacobian(result[-1], self.xs))
        return derivatives
