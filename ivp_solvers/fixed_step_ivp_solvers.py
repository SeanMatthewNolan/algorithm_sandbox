from abc import ABC

import numpy as np

from ivp_solvers import IVPSolver
import runge_kutta.explicit


class FixedStepRKSolver(IVPSolver, ABC):
    runge_kutta_stepper = runge_kutta.explicit.ExplicitRKStep()

    def __init__(self, step_size: float):
        self.step_size = step_size

    def solve(self, t0, tf, y0, dy_dt):
        t = [t0]
        y = [y0]

        while t[-1] <= tf:
            t.append(t[-1] + self.step_size)
            y.append(self.runge_kutta_stepper.step(t[-2], t[-1], y[-1], dy_dt))

        return np.array(t), np.array(y)


class CustomFixedStepRKSolver(FixedStepRKSolver):
    def __init__(self, step_size: float, stepper: runge_kutta.RKStep):
        super().__init__(step_size)
        self.runge_kutta_stepper = stepper


class FixedEuler(FixedStepRKSolver):
    runge_kutta_stepper = runge_kutta.explicit.Euler()


class FixedRK2(FixedStepRKSolver):
    runge_kutta_stepper = runge_kutta.explicit.Ralston()


class FixedRK4(FixedStepRKSolver):
    runge_kutta_stepper = runge_kutta.explicit.RK4()


class FixedThreeEigthsRK4(FixedStepRKSolver):
    runge_kutta_stepper = runge_kutta.explicit.ThreeEighthsRK4()
