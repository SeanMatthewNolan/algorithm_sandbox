from math import sin, cos, exp
from typing import Tuple, Callable

import pytest
import numpy as np

from ..jacobians import Jacobian, ForwardDiffJac

scalar_diff_funcs = [ForwardDiffJac]
examples = []


@pytest.mark.parametrize('example', examples)
@pytest.mark.parametrize('diff_class', scalar_diff_funcs)
def test_finite_step(diff_class: Jacobian, example: Tuple[Callable, Callable, float]):
    f, df_dx, x = example
    num_df_dx = diff_class(f)

    # print(num_df_dx(x), df_dx(x), num_df_dx(x) - df_dx(x))

    assert num_df_dx(x) == pytest.approx(df_dx(x), abs=1e-5, rel=1e-5)
