from abc import ABC, abstractmethod
from typing import Optional
from copy import copy

import numpy as np

from shared_utils.typing import ArrayFunction

# TODO: Look into ``numjac'' algorithm - ``The MATLAB ODE Suite Shampine''


class Jacobian(ABC):
    def __init__(self, func: ArrayFunction, arg_idx: int = 0):
        self.func = func
        self.arg_idx = arg_idx

    @abstractmethod
    def __call__(self, *args):
        pass


class ForwardDiffJac(Jacobian):
    def __init__(self, func: ArrayFunction, arg_idx: int = 0, step_size: float = 1e-6):
        super().__init__(func, arg_idx=arg_idx)
        self.step_size: float = step_size

    def __call__(self, *args, pre_calc_f: Optional[np.array] = None):
        if pre_calc_f is None:
            f_0 = self.func(*args)
        else:
            f_0 = pre_calc_f

        args_step = list(args)
        col_set = []
        for idx, _ in enumerate(args[self.arg_idx]):
            args_step[self.arg_idx] = copy(args[self.arg_idx])
            args_step[self.arg_idx][idx] += self.step_size
            col_set.append((self.func(*args_step) - f_0) / self.step_size)

        return np.vstack(col_set).T


class CentralDiffJac(Jacobian):
    def __init__(self, func: ArrayFunction, arg_idx: int = 0, step_size: float = 1e-6):
        super().__init__(func, arg_idx=arg_idx)
        self.step_size: float = step_size

    def __call__(self, *args):

        args_step_p, args_step_m = list(args), list(args)
        col_set = []
        for idx, _ in enumerate(args[self.arg_idx]):
            args_step_p[self.arg_idx], args_step_m[self.arg_idx] = copy(args[self.arg_idx]), copy(args[self.arg_idx])
            args_step_p[self.arg_idx][idx] += self.step_size / 2
            args_step_m[self.arg_idx][idx] -= self.step_size / 2

            col_set.append((self.func(*args_step_p) - self.func(*args_step_m)) / self.step_size)

        return np.vstack(col_set).T


def saddle(y):
    return np.array([y[0]**2 - y[1], -y[1]**2 + y[0]])


def multi_saddle(x, y):
    return np.array([x[0] * y[0]**2 - y[1], -x[1] * y[1]**2 + y[0]])


jac = CentralDiffJac(saddle)
print(jac(np.array([10., 10.])))

jac2 = CentralDiffJac(multi_saddle, arg_idx=1)
print(jac2(np.array([1., 1.]), np.array([10., 10.])))
