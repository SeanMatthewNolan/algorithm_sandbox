import numpy as np
import matplotlib.pyplot as plt

from ivp_solvers import IVP, IVPSolver


class PredatorPrey1(IVP):
    """
    Predator-prey taken from example 16.8 (pg. 496) of "A First Course in Numerical Methods" by Ascher and Greif
    """
    tf = 100
    y0 = np.array([80., 30.])
    yf = np.array([94.04588719, 38.11498521])  # For running tests

    def dy_dt(self, _, y):
        return np.array([
            0.25 * y[0] - 0.01 * y[0] * y[1],
            -y[1] + 0.01 * y[0] * y[1]
        ])


def plot_example(t, y, show=True):
    plt.plot(t, y)

    if show:
        plt.show()


def run_example(problem: IVP, solver: IVPSolver, plot=True, show=True):
    with Timer():
        t, y = solver.solve_problem(problem)

    if plot:
        plot_example(t, y, show=show)


if __name__ == '__main__':
    from runge_kutta.explicit import Midpoint
    from ivp_solvers import CustomFixedStepRKSolver, FixedEuler, FixedRK2, FixedRK4, FixedThreeEigthsRK4
    from shared_utils.tests import PerfTimerNS as Timer

    run_example(PredatorPrey1(), FixedEuler(step_size=0.01), show=False)
    run_example(PredatorPrey1(), CustomFixedStepRKSolver(0.1, Midpoint()), show=False)
    run_example(PredatorPrey1(), FixedRK2(step_size=0.1), show=False)
    run_example(PredatorPrey1(), FixedRK4(step_size=0.1), show=False)
    run_example(PredatorPrey1(), FixedThreeEigthsRK4(step_size=0.1), show=True)

    _t, _y = FixedThreeEigthsRK4(step_size=0.01).solve_problem(PredatorPrey1())

    plt.show()
