from abc import ABC

import numpy as np

from shared_utils.constants import EMPTY_ARRAY, EMPTY_ARRAY_2D


class RKStep(ABC):
    """
    Butcher - 2016 - NUMERICAL METHODS FOR ORDINARY DIFFERENTIAL EQUATIONS - pg 98

     c | A
    ----------
       | b^T

    A - dependence of the stages on the derivatives found at other stages
    b - vector of quadrature weights
    c - positions within step

    """

    a_mat = EMPTY_ARRAY_2D
    b = EMPTY_ARRAY

    def __init__(self):
        self.c = np.array(sum(a_j for a_j in self.a_mat.T))

    def stability_region_func(self, z: complex):
        """Butcher - 2016 - NUMERICAL METHODS FOR ORDINARY DIFFERENTIAL EQUATIONS - pg 98 """
        return 1. + z * np.sum(self.b @ np.linalg.inv(np.eye(*self.a_mat.shape) - z * self.a_mat))

    def plot_stability_region(self, n=150, max_len=10, max_root_steps=10):
        pass
