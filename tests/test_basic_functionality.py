# pylint: disable=redefined-outer-name
import importlib
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional

import numpy as np
import tensorflow as tf
from pytest_bdd import given, scenario, then, when

from xidiff import (XiDiffEquation, XiDiffModelWrapper, XiDiffSolver,
                    XiDiffVariables)


@dataclass
class EquationModuleContent:
    # pylint: disable=too-many-instance-attributes
    function_r: Callable[..., List[tf.Tensor]]
    function_i: Callable[..., List[tf.Tensor]]
    boundary_function: Callable[..., tf.Tensor]
    variable_ranges: List[Tuple[float, float]]
    number_of_functions: int
    evaluation_point_numpy: List
    evaluation_range_numpy: np.array
    evaluation_point_tensorflow: tf.Tensor
    evaluation_range_tensorflow: tf.Tensor


MODEL: Optional[XiDiffModelWrapper] = None


@scenario("features/basic_functionality.feature",
          "should approximate equation")
def test_should_approximate_equation() -> None:
    pass


@given(
    (
        "real part of equation, imaginary part of equation"
        ", boundary conditions and variable ranges"
    ),
    target_fixture="equation_data"
)
def equation_data() -> EquationModuleContent:
    module = importlib.import_module("tests.equations.equation_0")
    return EquationModuleContent(
        function_r=getattr(module, "function_r"),
        function_i=getattr(module, "function_i"),
        boundary_function=getattr(module, "boundary_function"),
        variable_ranges=getattr(module, "VARIABLE_RANGES"),
        number_of_functions=getattr(module, "NUMBER_OF_FUNCTIONS"),
        evaluation_point_numpy=getattr(module, "EVALUATION_POINT_NUMPY"),
        evaluation_range_numpy=getattr(module, "EVALUATION_RANGE_NUMPY"),
        evaluation_point_tensorflow=getattr(
            module, "EVALUATION_POINT_TENSORFLOW"),
        evaluation_range_tensorflow=getattr(
            module, "EVALUATION_RANGE_TENSORFLOW"),
    )


@given("xidiff variables are initialized", target_fixture="xidiff_variables")
def xidiff_variables(equation_data: EquationModuleContent) -> XiDiffVariables:
    return XiDiffVariables(equation_data.variable_ranges)


@given("xidiff equation is initialized", target_fixture="xidiff_equation")
def xidiff_equation(
    equation_data: EquationModuleContent,
    xidiff_variables: XiDiffVariables,
) -> XiDiffEquation:
    return XiDiffEquation(
        equation_data.function_r,
        equation_data.function_i,
        equation_data.boundary_function,
        equation_data.number_of_functions,
        xidiff_variables,
    )


@given("xidff solver is intia1ized", target_fixture="xidiff_solver")
def xidiff_solver(xidiff_equation: XiDiffEquation) -> XiDiffSolver:
    return XiDiffSolver(
        xidiff_equation, target_loss_exponent=-2
    )


@when("xidiff solver approximated the equation")
def xidiff_model(xidiff_solver: XiDiffSolver) -> None:
    #pylint: disable=global-statement
    global MODEL
    MODEL = xidiff_solver.approximate()


@then("it is possible to evaluate model")
def evaluate_model(
    equation_data: EquationModuleContent,
) -> None:
    #pylint: disable=global-statement
    global MODEL
    if MODEL is not None:
        MODEL(*equation_data.evaluation_point_numpy)
        MODEL(equation_data.evaluation_range_numpy)
        MODEL(equation_data.evaluation_point_tensorflow)
        MODEL(equation_data.evaluation_range_tensorflow)
