from typing import Optional

from shared_utils.typing import ScalarFunction, ArrayFunction, ArrayOrFloat
from ..root_solver_generic import RootSolver
from numeric_differentiation.finite_step import CentralDiff


class NewtonRaphsonScalar(RootSolver):
    def __init__(self, abs_tol: float = 1e-5, rel_tol: float = 1e-8, max_steps: int = 100):
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        self.max_steps = max_steps

    def __call__(self, func: ScalarFunction, guess: float, deriv_func: Optional[ScalarFunction] = None) -> float:
        if deriv_func is None:
            deriv_func = CentralDiff(func)

        x_n = guess
        f_n = func(x_n)
        tol = self.abs_tol + self.rel_tol * f_n

        for n in range(self.max_steps + 1):
            if abs(f_n) < tol:
                return x_n

            x_n -= f_n / deriv_func(x_n)
            f_n = func(x_n)

        # TODO: Add checks
        # TODO: Consolidate code for different versions


class DampedNewtonScalar(NewtonRaphsonScalar):
    def __init__(self, abs_tol: float = 1e-5, rel_tol: float = 1e-8, max_steps: int = 100, damp_coeff: float = 0.5):
        super().__init__(abs_tol=abs_tol, rel_tol=rel_tol, max_steps=max_steps)
        self.damp_coeff = damp_coeff

    def __call__(self, func: ScalarFunction, guess: float, deriv_func: Optional[ScalarFunction] = None) \
            -> float:
        if deriv_func is None:
            deriv_func = CentralDiff(func)

        x_n = guess
        f_n = func(x_n)
        tol = self.abs_tol + self.rel_tol * f_n

        for n in range(self.max_steps + 1):
            if abs(f_n) < tol:
                return x_n

            x_n -= self.damp_coeff * f_n / deriv_func(x_n)
            f_n = func(x_n)

        return x_n


class BoundedNewtonScalar(NewtonRaphsonScalar):
    def __init__(self, abs_tol: float = 1e-5, rel_tol: float = 1e-8, step_bound: float = 10.0):
        super().__init__(abs_tol=abs_tol, rel_tol=rel_tol)
        self.step_bound = step_bound

    def clip(self, raw_step):
        return min(max(raw_step, -self.step_bound), self.step_bound)

    def __call__(self, func: ScalarFunction, guess: float, deriv_func: Optional[ScalarFunction] = None) -> float:
        if deriv_func is None:
            deriv_func = CentralDiff(func)

        x_n = guess
        f_n = func(x_n)
        tol = self.abs_tol + self.rel_tol * f_n

        for n in range(self.max_steps + 1):
            if abs(f_n) < tol:
                return x_n

            x_n -= self.clip(f_n / deriv_func(x_n))
            f_n = func(x_n)

        return x_n


class NewtonArmijoScalar(NewtonRaphsonScalar):
    """
    [1] Armijo, L. “Minimization of Functions Having Lipschitz Continuous First Partial Derivatives.”
    Pacific Journal of Mathematics, Vol. 16, No. 1, 1966, pp. 1–3. https://doi.org/10.2140/pjm.1966.16.1.
    [2] Kelley, C. T. Iterative Methods for Linear and Nonlinear Equations. Society for Industrial and Applied
    Mathematics, Philadelphia, 1995.
    """

    def __init__(self, abs_tol: float = 1e-5, rel_tol: float = 1e-8, step_reduction_factor: int = 2, alpha: float = 1e-4):
        super().__init__(abs_tol=abs_tol, rel_tol=rel_tol)

        self.alpha = alpha
        self.step_reduction_ratio = 1 / step_reduction_factor

    def armijo_step(self, x_n, f_n, raw_step, func):
        lam = 1
        x_trial = x_n + raw_step
        f_trial = func(x_trial)
        abs_f_n = abs(f_n)

        while abs(f_trial) > (1 - self.alpha * lam) * abs_f_n:
            lam *= self.step_reduction_ratio
            x_trial = x_n + lam * raw_step
            f_trial = func(x_trial)

        return x_trial, f_trial

    def __call__(self, func: ScalarFunction, guess: float, deriv_func: Optional[ScalarFunction] = None) -> float:
        if deriv_func is None:
            deriv_func = CentralDiff(func)

        x_n = guess
        f_n = func(x_n)
        tol = self.abs_tol + self.rel_tol * f_n

        while abs(f_n) > tol:
            x_n, f_n = self.armijo_step(x_n, f_n, -f_n / deriv_func(x_n), func)

        return x_n
