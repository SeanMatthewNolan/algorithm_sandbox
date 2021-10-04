"""
Base Classes and Functions for Any BVP Solver Class
"""
from abc import ABC, abstractmethod

import numpy as np


class Solution:
    def __init__(self, t: np.array, y: np.array, sol_time: float = None, error: float = None):
        self.t = t
        self.y = y

        self.sol_time = sol_time
        self.error = error


class BVPSolver(ABC):
    """
    Base for all BVP solvers
    """
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def solve(self, dy_dt, bc_0, bc_f, guess):
        pass
