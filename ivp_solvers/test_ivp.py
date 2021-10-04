import pytest

from runge_kutta.explicit import Midpoint
from ivp_solvers import IVP
from ivp_solvers import FixedStepRKSolver, CustomFixedStepRKSolver, FixedEuler, FixedRK2, FixedRK4, FixedThreeEigthsRK4
from ivp_solvers.example_problems import PredatorPrey1


test_cases = [
    (PredatorPrey1(), FixedEuler(0.001)),
    (PredatorPrey1(), CustomFixedStepRKSolver(0.001, Midpoint())),
    (PredatorPrey1(), FixedRK2(0.01)),
    (PredatorPrey1(), FixedRK4(0.01)),
    (PredatorPrey1(), FixedThreeEigthsRK4(0.01))
]


@pytest.mark.parametrize('problem, solver', test_cases)
def test_fixed_step(problem: IVP, solver: FixedStepRKSolver):
    t, y = solver.solve_problem(problem)

    assert t[0] == problem.t0
    assert (t[-1] - problem.tf) <= solver.step_size
    assert y[0] == pytest.approx(problem.y0, rel=1e-6, abs=1e-6)
    assert y[-1] == pytest.approx(problem.yf, rel=1e-2, abs=1e-3)
