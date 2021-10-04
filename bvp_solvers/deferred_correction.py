import numpy as np

from bvp_generic import BVPSolver, Solution

machine_epsilon = np.finfo(float).eps


class DCMIRK(BVPSolver):
    """
    Algorithm based on
    Problems from "On the Numerical Integration of Nonlinear Two-Point Boundary Value Problems Using Deferred
    Corrections. Part 2: The Development and Analysis of Highly Stable Deferred Correction Formulae" by J. R. Cash
    """
    def __init__(self, tol=1e-4, rho=10 * machine_epsilon, mu=50, n_max=100, j_max=5):
        self.tau = tol
        self.rho = rho
        self.mu = mu

        self.n_max = 100
        self.j_max = 5

    def solve(self, dy_dt, bc_0, bc_f, guess: Solution, *args, **kwargs):
        hard = True
        j = 0
        au_max = 0

        t_n = guess.t
        eta = guess.y

        n_grid_pt = len(t_n)

        while n_grid_pt <= self.n_max:
            dc = -self.phi(eta)

