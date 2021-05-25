# pylint: disable=redefined-outer-name
import importlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

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
    order_of_system: int
    evaluation_point_numpy: List
    evaluation_range_numpy: np.array
    evaluation_point_tensorflow: tf.Tensor
    evaluation_range_tensorflow: tf.Tensor


MODEL: Optional[XiDiffModelWrapper] = None
MODEL_RESULTS: List = []
PATH_TO_MODEL = Path("model")


@scenario(
    "features/main_functionality.feature",
    "should approximate equation",
    example_converters={"equation": str},
)
def test_should_approximate_equation() -> None:
    pass


@given(
    (
        "real part, imaginary part, boundary conditions"
        " and variable ranges of <equation>"
    ),
    target_fixture="equation_data"
)
def equation_data(equation: str) -> EquationModuleContent:
    module = importlib.import_module(f"tests.equations.{equation}")
    return EquationModuleContent(
        function_r=getattr(module, "function_r"),
        function_i=getattr(module, "function_i"),
        boundary_function=getattr(module, "boundary_function"),
        variable_ranges=getattr(module, "VARIABLE_RANGES"),
        number_of_functions=getattr(module, "NUMBER_OF_FUNCTIONS"),
        order_of_system=getattr(module, "ORDER_OF_SYSTEM"),
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
        equation_data.order_of_system,
    )


@given("xidff solver is intia1ized", target_fixture="xidiff_solver")
def xidiff_solver(xidiff_equation: XiDiffEquation) -> XiDiffSolver:
    return XiDiffSolver(
        xidiff_equation, target_loss_exponent=-2,
    )


@when("xidiff solver approximated the equation")
def xidiff_model(xidiff_solver: XiDiffSolver) -> None:
    # pylint: disable=global-statement
    global MODEL
    assert MODEL is None
    MODEL = xidiff_solver.approximate()


@then("it is possible to evaluate model")
def evaluate_model(
    equation_data: EquationModuleContent,
) -> None:
    # pylint: disable=global-statement
    global MODEL
    global MODEL_RESULTS

    assert MODEL_RESULTS == []

    if MODEL is not None:
        MODEL_RESULTS.append(MODEL(*equation_data.evaluation_point_numpy))
        MODEL_RESULTS.append(MODEL(equation_data.evaluation_range_numpy))
        MODEL_RESULTS.append(MODEL(equation_data.evaluation_point_tensorflow))
        MODEL_RESULTS.append(MODEL(equation_data.evaluation_range_tensorflow))
    else:
        assert False, "MODEL must be present"


@then("it is possible to save model")
def save_model() -> None:
    # pylint: disable=global-statement
    global MODEL
    if MODEL is not None:
        MODEL.save(PATH_TO_MODEL)
        MODEL = None
    else:
        assert False, "MODEL must be present"


@then("it is possible to restore model")
def restore_model() -> None:
    # pylint: disable=global-statement
    global MODEL
    MODEL = XiDiffModelWrapper.load(PATH_TO_MODEL)


@then("it is possible to evaluate model with the same results")
def re_evaluate_model(equation_data: EquationModuleContent) -> None:
    # pylint: disable=global-statement
    global MODEL
    global MODEL_RESULTS
    results = []
    if MODEL is not None:
        results.append(MODEL(*equation_data.evaluation_point_numpy))
        results.append(MODEL(equation_data.evaluation_range_numpy))
        results.append(MODEL(equation_data.evaluation_point_tensorflow))
        results.append(MODEL(equation_data.evaluation_range_tensorflow))
        assert (results[0] == MODEL_RESULTS[0]).all()
        assert (results[1] == MODEL_RESULTS[1]).all()
        assert (results[2] == MODEL_RESULTS[2]).all()
        assert (results[3] == MODEL_RESULTS[3]).all()
    else:
        assert False, "MODEL must be present"


@then("clean up")
def clean_up() -> None:
    # pylint: disable=global-statement
    global MODEL
    global MODEL_RESULTS

    shutil.rmtree(PATH_TO_MODEL)
    MODEL = None
    MODEL_RESULTS = []
