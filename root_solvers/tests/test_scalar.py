from math import sin, exp
from itertools import product
from typing import Tuple, Callable

import pytest

from ..root_solver_generic import RootSolver
from .. import NewtonRaphsonScalar, DampedNewtonScalar, BoundedNewtonScalar
from ..scalar.newton_raphson import NewtonArmijoScalar

abs_tol = 1e-5
rel_tol = 1e-8

scalar_root_solvers = [
    NewtonRaphsonScalar(abs_tol=abs_tol, rel_tol=rel_tol),
    DampedNewtonScalar(abs_tol=abs_tol, rel_tol=rel_tol),
    BoundedNewtonScalar(abs_tol=abs_tol, rel_tol=rel_tol),
    NewtonArmijoScalar(abs_tol=abs_tol, rel_tol=rel_tol)
]

examples = [
    (sin, 3.14159/4),
    (lambda x: exp(x) - 1, -2),
    (lambda x: x**3 - x, 10),
]


@pytest.mark.parametrize('root_solver', scalar_root_solvers)
@pytest.mark.parametrize('example', examples)
def test_scalar_root_solver(root_solver: RootSolver, example: Tuple[Callable, float]):
    f, x_guess = example
    x_star = root_solver(f, x_guess)

    tol = abs_tol + rel_tol * f(x_guess)
    assert f(x_star) == pytest.approx(0, abs=tol)
