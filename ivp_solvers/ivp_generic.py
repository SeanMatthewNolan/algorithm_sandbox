from abc import ABC, abstractmethod

from shared_utils.constants import EMPTY_ARRAY


class IVP(ABC):
    t0 = 0
    tf = 1
    y0 = EMPTY_ARRAY
    yf = EMPTY_ARRAY

    @abstractmethod
    def dy_dt(self, t, y):
        pass


class IVPSolver(ABC):
    """
    Base for all BVP solvers
    """

    @abstractmethod
    def solve(self, t0, t_f, y0, dy_dt):
        pass

    def solve_problem(self, problem: IVP):
        return self.solve(problem.t0, problem.tf, problem.y0, problem.dy_dt)

