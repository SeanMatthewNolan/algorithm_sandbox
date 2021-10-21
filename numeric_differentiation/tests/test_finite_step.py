from math import sin, cos, exp
from itertools import product
from typing import Tuple, Callable

import pytest
import numpy as np

from ..finite_step import NumDiff, ForwardDiff, BackwardDiff, CentralDiff

scalar_diff_funcs = [ForwardDiff, BackwardDiff, CentralDiff]
examples = [
    (sin, cos, 0),
    (cos, lambda x: -sin(x), np.pi / 4),
    (exp, exp, -1),
    (lambda x: x**3 - 2*x**2 + x, lambda x: 3*x**2 - 4*x + 1, 1)
]


@pytest.mark.parametrize('diff_class, example', product(scalar_diff_funcs, examples))
def test_finite_step(diff_class: NumDiff, example: Tuple[Callable, Callable, float]):
    f, df_dx, x = example
    num_df_dx = diff_class(f)

    # print(num_df_dx(x), df_dx(x), num_df_dx(x) - df_dx(x))

    assert num_df_dx(x) == pytest.approx(df_dx(x), abs=1e-5, rel=1e-5)
