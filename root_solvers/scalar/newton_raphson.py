from typing import Optional

from shared_utils.typing import ScalarFunction, ArrayFunction, ArrayOrFloat
from ..root_solver_generic import RootSolver
from numeric_differentiation.finite_step import CentralDiff


class NewtonRaphsonScalar(RootSolver):
    def __init__(self, tol=1e-5, max_steps=100):
        self.tol = tol
        self.max_steps = max_steps

    def __call__(self, func: ScalarFunction, guess: float, deriv_func: Optional[ScalarFunction] = None) \
            -> float:
        if deriv_func is None:
            deriv_func = CentralDiff(func)

        x_n = guess
        for step_num in range(self.max_steps):
            f_n = func(x_n)
            if abs(f_n) < self.tol:
                break

            x_n -= f_n / deriv_func(x_n)

        return x_n

