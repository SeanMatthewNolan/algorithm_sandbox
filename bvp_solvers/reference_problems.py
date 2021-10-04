"""
File of Referrence BVP Problems
"""
from abc import ABC, abstractmethod

import numpy as np


class BVP(ABC):
    """
    Base Class for Reference BVP Problems
    """

    @abstractmethod
    def dy_dt(self, t: float, y: np.array) -> np.array:
        """
        Differential equation of BVP

        :param t: time
        :param y: state vector
        :return: derivative vector
        """
        pass

    @abstractmethod
    def bc_0(self, t: float, y: np.array) -> np.array:
        """
        Initial boundary condition vector

        :param t: time
        :param y: state vector
        :return: residual vector
        """
        pass

    @abstractmethod
    def bc_f(self, t: float, y: np.array) -> np.array:
        """
        Terminal boundary condition vector

        :param t: time
        :param y: state vector
        :return: residual vector
        """
        pass


"""
Problems from "On the Numerical Integration of Nonlinear Two-Point Boundary Value Problems Using Deferred Corrections. 
Part 2: The Development and Analysis of Highly Stable Deferred Correction Formulae" by J. R. Cash
"""


class CashProblem1(BVP):
    """
    Problem 1 from above paper
    Difficulty increases with decreasing epsilon

    eps*y'' - y' = 0
    y(0) = 1; y(1) = 2
    """
    def __init__(self, epsilon=-0.1):
        """
        Initialization of CashProblem1

        :param epsilon: 'difficulty parameter'
        """
        self.epsilon = epsilon

    def dy_dt(self, t: float, y: np.array) -> np.array:
        return np.array([y[1], y[1] / self.epsilon])

    def bc_0(self, t: float, y: np.array) -> np.array:
        return np.array([t, y[0] - 1])

    def bc_f(self, t: float, y: np.array) -> np.array:
        return np.array([t - 1, y[0] - 2])
