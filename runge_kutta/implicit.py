from abc import ABC, abstractmethod

import numpy as np

from runge_kutta import RKStep
from shared_utils.constants import EMPTY_ARRAY, EMPTY_ARRAY_2D


class ImplicitRK(RKStep, ABC):
    def compute_error(self, t, y, f):
        pass


class MIRK(ImplicitRK, ABC):
    v = EMPTY_ARRAY
    x_mat = EMPTY_ARRAY_2D

    def __init__(self):
        self.a_mat = self.x_mat + np.outer(self.v, self.b)
        super().__init__()

    def compute_error(self, t, y, f):
        h = t[-1] - t[0]

        k = []
        diff_sum = 0
        for b_r, c_r, v_r, x_r in zip(self.b, self.c, self.v, self.x_mat):
            t_r = t[0] + h * c_r
            y_r = (1 - v_r) * y[0] + v_r * y[-1] + h * sum(x_rj * k_j for x_rj, k_j in zip(x_r, k))
            k.append(f(t_r, y_r))
            diff_sum += b_r * k[-1]

        return y[0] + h * diff_sum - y[-1]


class MIRK1(MIRK):
    b = np.array([1.])
    v = np.array([0.])
    x_mat = np.array([[0.]])


class MIRK21(MIRK):
    b = np.array([1.])
    v = np.array([0.5])
    x_mat = np.array([[0.]])


class MIRK22(MIRK):
    b = np.array([0.5, 0.5])
    v = np.array([0., 1.])
    x_mat = np.array([[0., 0.], [0., 0.]])


class MIRK32(MIRK):
    b = np.array([0.25, 0.75])
    v = np.array([1., 5/9])
    x_mat = np.array([[0., 0.], [-2/9, 0.]])


class MIRK43(MIRK):
    b = np.array([1/6, 1/6, 2/3])
    v = np.array([0., 1., 0.5])
    x_mat = np.array([[0., 0., 0.], [0., 0., 0.], [1/8, -1/8, 0]])


class MIRK65(MIRK):
    from math import sqrt
    b = np.array([1/20, 1/20, 49/180, 49/180, 16/45])
    v = np.array([0., 1., 1/2 - 9 * sqrt(21) / 98, 1/2 + 9 * sqrt(21) / 98, 1/2])
    x_mat = np.array(
            [
                [0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0.],
                [1 / 14 + sqrt(21) / 98, -1 / 14 + sqrt(21) / 98, 0., 0., 0.],
                [1 / 14 - sqrt(21) / 98, -1 / 14 - sqrt(21) / 98, 0., 0., 0.],
                [-5 / 128, 5 / 128, 7 * sqrt(21) / 128, -7 * sqrt(21) / 128, 0.],
            ])


if __name__ == '__main__':
    import scipy.optimize
    import scipy.integrate

    class Problem(ABC):
        @abstractmethod
        def f(self, _, y):
            pass

        @abstractmethod
        def psi(self, y_0, y_f):
            pass

        @abstractmethod
        def analytical_sol(self, t):
            pass

        def vect_f(self, t, y):
            return np.array([self.f(t_i, y_i) for t_i, y_i in zip(t, y.T)]).T

    class Problem0(Problem):
        def __init__(self):
            self.tspan = (0., 1)
            self.k0 = -1.
            self.k1 = 1.
            self.k2 = -1.
            self.k3 = 1.

        def f(self, _, y):
            return np.array([y[1], y[2], self.k3])

        def psi(self, y_0, y_f):
            return np.array([y_0[0] - self.k0, y_0[1] - self.k1,  y_f[0] - (self.analytical_sol(self.tspan[-1]))])

        def analytical_sol(self, t):
            return self.k3/6*t**3 + self.k2/2*t**2 + self.k1*t + self.k0

    class Problem2(Problem):
        """
        Cash and Wright 1991
        """
        def __init__(self):
            self.alpha = 1
            self.beta = 0.745
            self.eps = 1

        def f(self, _, y):
            return np.array([y[1], (1 - y[1]**2)/self.eps])

        def psi(self, y_0, y_f):
            pass

        def analytical_sol(self, t):
            return self.alpha + self.eps * np.log((t - self.beta) / self.eps)


    class Problem3(Problem):
        """
        Cash and Wright 1991
        """
        def __init__(self):
            self.eps = 1
            self.tspan = (0., 1.)

        def f(self, _, y):
            return np.array([y[1], (1 - y[1]**2)/self.eps])

        def psi(self, y_0, y_f):
            return np.array([y_0[0] - self.analytical_sol(self.tspan[0]), y_f[0] - self.analytical_sol(self.tspan[1])])

        def analytical_sol(self, t):
            return np.exp(-t/self.eps)


    mirk21 = MIRK21()
    mirk32 = MIRK32()
    mirk43 = MIRK43()
    mirk65 = MIRK65()

    prob = Problem3()

    def nlp_func21(x):
        return mirk21.compute_error(
                prob.tspan,
                (np.array([prob.analytical_sol(prob.tspan[0]), x[0]]),
                 np.array([prob.analytical_sol(prob.tspan[1]), x[1]])),
                prob.f)


    def nlp_func32(x):
        return mirk32.compute_error(
                prob.tspan,
                (np.array([prob.analytical_sol(prob.tspan[0]), x[0]]),
                 np.array([prob.analytical_sol(prob.tspan[1]), x[1]])),
                prob.f)


    def nlp_func43(x):
        return mirk43.compute_error(
                prob.tspan,
                (np.array([prob.analytical_sol(prob.tspan[0]), x[0]]),
                 np.array([prob.analytical_sol(prob.tspan[1]), x[1]])),
                prob.f)


    def nlp_func65(x):
        return mirk65.compute_error(
                prob.tspan,
                (np.array([prob.analytical_sol(prob.tspan[0]), x[0]]),
                 np.array([prob.analytical_sol(prob.tspan[1]), x[1]])),
                prob.f)


    x_star21 = scipy.optimize.fsolve(nlp_func21, np.array([0., 0.]))
    print(x_star21)
    # print(nlp_func21(x_star21))

    x_star32 = scipy.optimize.fsolve(nlp_func32, np.array([0., 0.]))
    print(x_star32)
    # print(nlp_func32(x_star32))

    x_star43 = scipy.optimize.fsolve(nlp_func43,  np.array([0., 0.]))
    print(x_star43)
    # print(nlp_func43(x_star43))

    x_star65 = scipy.optimize.fsolve(nlp_func65,  np.array([0., 0.]))
    print(x_star65)
    # print(nlp_func65(x_star65))

    sol = scipy.integrate.solve_bvp(
            prob.vect_f, prob.psi, prob.tspan,
            np.array([[prob.analytical_sol(prob.tspan[0]), -1.], [prob.analytical_sol(prob.tspan[1]), -0.3]]))

    print(sol.y[1, 0], sol.y[1, -1])

