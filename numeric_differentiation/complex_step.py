from .finite_step import NumDiff
from shared_utils.typing import ComplexScalarFunction


class ComplexStep(NumDiff):
    def __init__(self,  func: ComplexScalarFunction, step_size: float = 1e-100):
        super().__init__(func, step_size)

    def __call__(self, value: complex):
        return self.func(value + self.step_size * 1j).imag / self.step_size
