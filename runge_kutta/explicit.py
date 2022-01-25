from abc import ABC

import numpy as np

from runge_kutta import RKStep


class ExplicitRKStep(RKStep, ABC):
    def compute_step(self, t_0, t_f, y_0, f):
        h = t_f - t_0

        step = 0
        t_arr = []
        y_arr = []
        for j, (a_j, b_j, c_j) in enumerate(zip(self.a_mat, self.b, self.c)):
            t_j = t_0 + h * c_j
            y_j = y_0

            for k, (t_k, y_k, a_jk) in enumerate(zip(t_arr, y_arr, a_j)):
                y_j = y_j + h * a_jk * f(t_k, y_k)

            t_arr.append(t_j)
            y_arr.append(y_j)
            step += h * b_j * f(t_j, y_j)

        return step

    def step(self, t_0, t_f, y_0, f):
        return y_0 + self.compute_step(t_0, t_f, y_0, f)


"""1st Order"""


class Euler(ExplicitRKStep):
    a_mat = np.array([[0.]])
    b = np.array([1.])


"""2nd Order"""


class Midpoint(ExplicitRKStep):
    """
    pg. 499 of "A First Course in Numerical Methods" by Ascher and Greif
    """
    a_mat = np.array([
        [0.,  0.],
        [1/2, 0.]
    ])
    b = np.array([0., 1])


class Heun(ExplicitRKStep):
    """
    https://en.wikipedia.org/wiki/Heun%27s_method#Runge.E2.80.93Kutta_method
    """
    a_mat = np.array([
        [0., 0.],
        [1., 0.]
    ])
    b = np.array([1/2, 1/2])


class Ralston(ExplicitRKStep):
    """
    https://en.wikipedia.org/wiki/Heun%27s_method#Runge.E2.80.93Kutta_method
    Minimizes truncation error
    """
    a_mat = np.array([
        [0., 0.],
        [2/3., 0.]
    ])
    b = np.array([1/4, 3/4])


"""4th Order"""


class RK4(ExplicitRKStep):
    a_mat = np.array([
        [0.,  0.,  0., 0.],
        [1/2, 0.,  0., 0.],
        [0.,  1/2, 0., 0.],
        [0.,  0.,  1., 0.]
    ])
    b = np.array([1/6, 1/3, 1/3, 1/6])


class ThreeEighthsRK4(ExplicitRKStep):
    a_mat = np.array([
        [0.,    0., 0., 0.],
        [1/3,   0., 0., 0.],
        [-1/3,  1., 0., 0.],
        [1.,   -1., 1., 0.]
    ])
    b = np.array([1/8, 3/8, 3/8, 1/8])
