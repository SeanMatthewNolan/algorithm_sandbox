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


class DampedNewtonScalar(RootSolver):
    def __init__(self, tol=1e-5, max_steps=100, damp_coeff=0.5):
        self.tol = tol
        self.max_steps = max_steps
        self.damp_coeff = damp_coeff

    def __call__(self, func: ScalarFunction, guess: float, deriv_func: Optional[ScalarFunction] = None) \
            -> float:
        if deriv_func is None:
            deriv_func = CentralDiff(func)

        x_n = guess
        for step_num in range(self.max_steps):
            f_n = func(x_n)
            if abs(f_n) < self.tol:
                break

            x_n -= self.damp_coeff * f_n / deriv_func(x_n)

        return x_n


class BoundedNewtonScalar(RootSolver):
    def __init__(self, tol=1e-5, max_steps=100, step_bound=10.0):

        self.tol = tol
        self.max_steps = max_steps
        self.step_bound = step_bound

    from numpy import clip as np_clip

    def clip(self, raw_step):
        return min(max(raw_step, -self.step_bound), self.step_bound)

    def __call__(self, func: ScalarFunction, guess: float, deriv_func: Optional[ScalarFunction] = None) -> float:
        if deriv_func is None:
            deriv_func = CentralDiff(func)

        x_n = guess
        for step_num in range(self.max_steps):
            f_n = func(x_n)
            if abs(f_n) < self.tol:
                break

            x_n -= self.clip(f_n / deriv_func(x_n))

        return x_n
