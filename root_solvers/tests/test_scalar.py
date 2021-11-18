from cmath import sin, cos, exp
from itertools import product
from typing import Tuple, Callable

import pytest
import numpy as np

from ..root_solver_generic import RootSolver
from .. import NewtonRaphsonScalar

tol = 1e-5

scalar_root_solvers = [
    NewtonRaphsonScalar(tol=tol)
]

examples = [
    (sin, 3.14159/4),
    (lambda x: exp(x), -2),
    (lambda x: x**3 - x, 10),
]


@pytest.mark.parametrize('root_solver, example', product(scalar_root_solvers, examples))
def test_complex_step(root_solver: RootSolver, example: Tuple[Callable, float]):
    f, x_guess = example
    x_star = root_solver(f, x_guess)

    assert f(x_star) == pytest.approx(0, abs=tol)
