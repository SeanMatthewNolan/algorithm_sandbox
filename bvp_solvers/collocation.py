import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

from bvp_generic import BVPSolver, Solution
from numeric_differentiation.jacobians import CentralDiffJac
from shared_utils.typing import DiffEqn

SPARSE = False


class Collocation4:
    def __init__(self, f: DiffEqn, g, abs_tol: float = 1e-5, rel_tol: float = 1e-5):
        self.f = f
        self.g = g

        self.df_dy = CentralDiffJac(self.f, arg_idx=1)
        self.dg_dy0 = CentralDiffJac(self.g, arg_idx=0)
        self.dg_dyf = CentralDiffJac(self.g, arg_idx=1)

        self.abs_tol = abs_tol
        self.rel_tol = rel_tol

        self._n = None
        self._identity = None

        self.max_steps = 21
        self.alpha = 1e-4
        self.step_reduction_ratio = 0.5

    def armijo_step(self, x_n, f_n, raw_step, func):
        lam = 1
        x_trial = x_n + raw_step
        f_trial = func(x_trial)
        abs_f_n = abs(f_n)

        while abs(f_trial) > (1 - self.alpha * lam) * abs_f_n:
            lam *= self.step_reduction_ratio
            x_trial = x_n + lam * raw_step
            f_trial = func(x_trial)

        return x_trial, f_trial

    def compute_phi(self, t_set: np.ndarray, y_set: np.ndarray):
        f_set = np.array([self.f(t_i, y_i) for t_i, y_i in zip(t_set, y_set)])

        phi_set = [self.g(y_set[0], y_set[-1])]
        zipped = list(zip(t_set, y_set, f_set))
        for (t_low, y_low, f_low), (t_high, y_high, f_high) in zip(zipped[:-1], zipped[1:]):
            h = t_high - t_low
            f_mid = self.f(t_low + h / 2, (y_low + y_high) / 2 - h / 8 * (f_high - f_low))
            phi_set.append(y_high - y_low - h / 6 * (f_low + 4 * f_mid + f_high))

        return np.concatenate(phi_set)

    def compute_phi_and_diff(self, t_set: np.ndarray, y_set: np.ndarray):
        f_set = np.array([self.f(t_i, y_i) for t_i, y_i in zip(t_set, y_set)])
        jac_set = np.array([self.df_dy(t_i, y_i) for t_i, y_i in zip(t_set, y_set)])

        phi_set = [self.g(y_set[0], y_set[-1])]
        dphi_dy_upper = [self.dg_dy0(y_set[0], y_set[-1])]
        dphi_dy_lower = []

        zipped = list(zip(t_set, y_set, f_set, jac_set))
        for (t_low, y_low, f_low, jac_low), (t_high, y_high, f_high, jac_high) in zip(zipped[:-1], zipped[1:]):
            h = t_high - t_low
            t_mid, y_mid = t_low + h / 2, (y_low + y_high) / 2 - h / 8 * (f_high - f_low)

            f_mid = self.f(t_mid, y_mid)
            jac_half = self.df_dy(t_mid, y_mid)

            phi_set.append(y_high - y_low - h / 6 * (f_low + 4 * f_mid + f_high))
            dphi_dy_upper.append(
                    self._identity - 1/6 * h * jac_high
                    - 2/3 * h * jac_half * (1/2 * self._identity - 1/8 * h * jac_high))
            dphi_dy_lower.append(
                    -self._identity - 1/6 * h * jac_low
                    - 2/3 * h * jac_half * (1/2 * self._identity + 1/8 * h * jac_low))

        if SPARSE:
            dphi_dy = scipy.sparse.block_diag(dphi_dy_upper, format='lil')
            dphi_dy[self._n:, :-self._n] += scipy.sparse.block_diag(dphi_dy_lower, format='lil')
            dphi_dy[:self._n, -self._n:] = self.dg_dyf(y_set[0], y_set[-1])
            dphi_dy = scipy.sparse.csr_matrix(dphi_dy)
        else:
            dphi_dy = scipy.linalg.block_diag(*dphi_dy_upper)
            dphi_dy[self._n:, :-self._n] += scipy.linalg.block_diag(*dphi_dy_lower)
            dphi_dy[:self._n, -self._n:] = self.dg_dyf(y_set[0], y_set[-1])

        return np.concatenate(phi_set), dphi_dy

    def solve(self, guess: Solution):
        if self._identity is None:
            self._n = len(guess.y[0])
            self._identity = np.identity(self._n, dtype=float)

        t_n, y_n = guess.t, guess.y
        phi_n = self.compute_phi(t_n, y_n)

        tol = self.abs_tol + self.rel_tol * phi_n
        for iteration in range(self.max_steps):
            if all(np.abs(phi_n) < tol):
                break

            phi_n, dphi_n = self.compute_phi_and_diff(t_n, y_n)

            if SPARSE:
                raw_step = scipy.sparse.linalg.spsolve(dphi_n, -phi_n).reshape(y_n.shape)
            else:
                try:
                    raw_step = scipy.linalg.solve(dphi_n, -phi_n).reshape(y_n.shape)
                except np.linalg.LinAlgError:
                    raw_step = (scipy.linalg.pinv(dphi_n) @ -phi_n).reshape(y_n.shape)

            y_trial = y_n + raw_step
            phi_trial = self.compute_phi(t_n, y_trial)
            norm_phi_n = scipy.linalg.norm(phi_n)

            lam = 1
            while scipy.linalg.norm(phi_trial) > (1 - self.alpha * lam) * norm_phi_n:
                lam *= self.step_reduction_ratio
                y_trial = y_n + lam * raw_step
                phi_trial = self.compute_phi(t_n, y_trial)

            y_n, phi_n = y_trial, phi_trial

        sol = Solution(t_n, y_n)
        return sol


if __name__ == '__main__':
    import math
    import matplotlib.pyplot as plt

    def f_(_, y):
        return np.array([y[1], -y[0]])


    def g_(ya, yb):
        return np.array([ya[0] - 1, yb[1] - 1])


    _guess = Solution(np.linspace(0, 1, 10), np.ones((10, 2)))
    _sol = Collocation4(f_, g_).solve(_guess)

    def measels_f(x, y):
        beta = 1575*(1 + math.cos(2*3.14159*x))
        return np.array([
            0.02 - beta*y[0]*y[2],
            beta*y[0]*y[2] - y[1]/0.0279,
            y[1]/0.0279 - y[2]/0.01
        ])

    def measels_g(ya, yb):
        return ya - yb

    n_steps = 21
    # measels_guess = Solution(
    #         np.linspace(0, 1, n_steps), np.hstack((np.ones((n_steps, 1))*0.01, np.zeros((n_steps, 2)))))
    measels_guess = Solution(np.linspace(0, 1, n_steps), np.ones((n_steps, 3)) * 0.01)
    measels_sol = Collocation4(measels_f, measels_g).solve(measels_guess)

    plt.plot(measels_sol.t, measels_sol.y[:, 0])
    plt.plot(measels_sol.t, measels_sol.y[:, 1:])
    plt.show()
