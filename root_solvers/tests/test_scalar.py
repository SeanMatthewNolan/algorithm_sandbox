from math import sin, exp
from itertools import product
from typing import Tuple, Callable

import pytest

from ..root_solver_generic import RootSolver
from .. import NewtonRaphsonScalar, DampedNewtonScalar, BoundedNewtonScalar

tol = 1e-5

scalar_root_solvers = [
    NewtonRaphsonScalar(tol=tol),
    DampedNewtonScalar(tol=tol),
    BoundedNewtonScalar(tol=tol)
]

examples = [
    (sin, 3.14159/4),
    (lambda x: exp(x) - 1, -2),
    (lambda x: x**3 - x, 10),
]


@pytest.mark.parametrize('root_solver, example', product(scalar_root_solvers, examples))
def test_scalar_root_solver(root_solver: RootSolver, example: Tuple[Callable, float]):
    f, x_guess = example
    x_star = root_solver(f, x_guess)

    assert f(x_star) == pytest.approx(0, abs=tol)
