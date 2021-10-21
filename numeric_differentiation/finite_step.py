from abc import ABC, abstractmethod

from shared_utils.typing import ScalarFunction


class NumDiff(ABC):
    def __init__(self, func: ScalarFunction, step_size: float = 1e-6):
        self.func = func
        self.step_size = step_size

    @abstractmethod
    def __call__(self, value: float):
        pass


class ForwardDiff(NumDiff):
    def __call__(self, value: float):
        return (self.func(value + self.step_size) - self.func(value)) / self.step_size


class BackwardDiff(NumDiff):
    def __call__(self, value: float):
        return (self.func(value) - self.func(value - self.step_size)) / self.step_size


class CentralDiff(NumDiff):
    def __call__(self, value: float):
        return (self.func(value + self.step_size / 2) - self.func(value - self.step_size / 2)) / self.step_size
