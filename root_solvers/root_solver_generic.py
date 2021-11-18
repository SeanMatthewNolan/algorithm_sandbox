"""
Base Classes and Functions for Any Root Solver Class
"""
from abc import ABC, abstractmethod
from shared_utils.typing import ArrayOrFloat, Function


class RootSolver(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, func: Function, guess: ArrayOrFloat) -> ArrayOrFloat:
        pass
