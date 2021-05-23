from __future__ import annotations

import inspect
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Type

import tensorflow as tf
from mypy_extensions import VarArg

if TYPE_CHECKING:
    from xidiff.variables import XiDiffVariables


XiDiffFunction = Callable[[VarArg(tf.Tensor)], List[tf.Tensor]]


class XiDiffEquation:
    def __init__(
        # pylint: disable=too-many-arguments
        self: XiDiffEquation,
        real_function: XiDiffFunction,
        imaginary_function: XiDiffFunction,
        boundary_conditions_function: XiDiffFunction,
        unknown_functions: int,
        variables: XiDiffVariables,
    ) -> None:
        self.real_function = real_function
        self.imaginary_function = imaginary_function
        self.boundary_conditions_function = boundary_conditions_function
        self.unknown_functions = unknown_functions
        self.variables = variables
        self.highest_order_derivative = self._init_highest_order_derivative()

    def _init_highest_order_derivative(self: XiDiffEquation) -> int:
        arguments_length = len(inspect.getfullargspec(self.real_function).args)
        # first parameter of the python function is tensor of arguments
        # second parameter is the function value
        # third and onwards are tensors of derivatives
        return arguments_length - 2

    def save(
        self: XiDiffEquation,
        dir_path: Path,
    ) -> None:
        path = dir_path / "model.eqn"

        with open(path, "wb") as file_handler:
            pickle.dump(self, file_handler)

    @classmethod
    def load(
        cls: Type[XiDiffEquation],
        dir_path: Path,
    ) -> None:
        path = dir_path / "model.eqn"

        with open(path, "rb") as file_handler:
            return pickle.load(file_handler)
