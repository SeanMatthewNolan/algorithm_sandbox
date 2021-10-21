from typing import Callable, Union

import numpy as np

ArrayOrFloat = Union[np.ndarray, float]

Function = Callable[[ArrayOrFloat], ArrayOrFloat]
ScalarFunction = Callable[[float], float]
ComplexScalarFunction = Callable[[complex], complex]

ArrayFunction = Callable[[np.ndarray], np.ndarray]
DiffEqn = Callable[[float, ArrayOrFloat], ArrayOrFloat]
