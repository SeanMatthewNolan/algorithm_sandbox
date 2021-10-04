from abc import ABC

import numpy as np

from shared_utils.constants import EMPTY_ARRAY, EMPTY_ARRAY_2D


class RKStep(ABC):
    a = EMPTY_ARRAY_2D
    b = EMPTY_ARRAY

    def __init__(self):
        self.c = np.array(sum(a_j for a_j in self.a.T))
