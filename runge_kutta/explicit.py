from abc import ABC

import numpy as np

from runge_kutta import RKStep


class ExplicitRKStep(RKStep, ABC):
    """
    Butcher - 2016 - NUMERICAL METHODS FOR ORDINARY DIFFERENTIAL EQUATIONS - pg 98

     c | A
    ----------
       | b^T

    A - dependence of the stages on the derivatives found at other stages
    b - vector of quadrature weights
    c - positions within step

    """
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


class GenericSecondOrder(ExplicitRKStep):
    """Butcher - 2016 - NUMERICAL METHODS FOR ORDINARY DIFFERENTIAL EQUATIONS - pg 99"""
    def __init__(self, theta):
        self.theta = theta

        self.a_mat = np.array([[0., 0.], [self.theta, 0.]])
        self.b = np.array([1 - 1/(2*self.theta), 1/(2*self.theta)])
        super().__init__()


"""3rd Order"""


class RK31(ExplicitRKStep):
    """Butcher - 2016 - NUMERICAL METHODS FOR ORDINARY DIFFERENTIAL EQUATIONS - pg 99"""
    a_mat = np.array([
        [0., 0., 0.],
        [2/3, 0., 0.],
        [1/3, 1/3, 0.],
    ])
    b = np.array([1/4, 0., 3/4])


class RK32(ExplicitRKStep):
    """Butcher - 2016 - NUMERICAL METHODS FOR ORDINARY DIFFERENTIAL EQUATIONS - pg 99"""
    a_mat = np.array([
        [0., 0., 0.],
        [1/2, 0., 0.],
        [-1, 2., 0.],
    ])
    b = np.array([1/6, 2/3, 3/6])


"""4th Order"""


class RK4(ExplicitRKStep):
    a_mat = np.array([
        [0.,  0.,  0., 0.],
        [1/2, 0.,  0., 0.],
        [0.,  1/2, 0., 0.],
        [0.,  0.,  1., 0.]
    ])
    b = np.array([1/6, 1/3, 1/3, 1/6])


class RK42(ExplicitRKStep):
    """Butcher - 2016 - NUMERICAL METHODS FOR ORDINARY DIFFERENTIAL EQUATIONS - pg 102"""
    a_mat = np.array([
        [0.,  0.,  0., 0.],
        [1/4, 0.,  0., 0.],
        [0.,  1/2, 0., 0.],
        [1.,  -2., 2., 0.]
    ])
    b = np.array([1/6, 0, 2/3, 1/6])


class ThreeEighthsRK4(ExplicitRKStep):
    a_mat = np.array([
        [0.,    0., 0., 0.],
        [1/3,   0., 0., 0.],
        [-1/3,  1., 0., 0.],
        [1.,   -1., 1., 0.]
    ])
    b = np.array([1/8, 3/8, 3/8, 1/8])


"""5th Order"""


class RK5(ExplicitRKStep):
    """Butcher - 2016 - NUMERICAL METHODS FOR ORDINARY DIFFERENTIAL EQUATIONS - pg 103"""
    a_mat = np.array([
        [0.,  0.,  0., 0., 0., 0.],
        [1/4, 0.,  0., 0., 0., 0.],
        [1/8, 1/8, 0., 0., 0., 0.],
        [0., 0., 1/2, 0., 0., 0.],
        [3/16, -3/8, 3/8, 9/16, 0., 0.],
        [-3/7, 8/7, 6/7, -12/7, 8/7, 0.],
    ])
    b = np.array([7/90, 0., 32/90, 12/90, 32/90, 7/90])


if __name__ == '__main__':
    rk4 = RK4()
    rk4.stability_region_func(1)

