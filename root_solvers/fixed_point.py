from shared_utils.typing import Function, ArrayOrFloat


class FixedPointSolver:
    def __init__(self, max_iter=500, abs_tol=1e-4, rel_tol=1e-4):
        self.max_iter = max_iter
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol

    def __call__(self, guess: ArrayOrFloat, func: Function) -> ArrayOrFloat:
        eta0 = guess
        for i in range(self.max_iter):
            tol = self.abs_tol + self.rel_tol * eta0
            eta1 = func(eta0)
            if abs(eta1 - eta0) < tol:
                return eta1

            eta0 = eta1

        else:
            print('Max iteration reached without satisfying tolerances')
            return eta0
