from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np


class Classifier(ABC):
    """Abstract class for change point detection algorithms"""

    @property
    @abstractmethod
    def window(self) -> list[float | np.float64] | None:
        raise NotImplementedError
    
    @window.setter
    @abstractmethod
    def window(self, val: Iterable[float | np.float64]) -> None:
        raise NotImplementedError

    @abstractmethod
    def classify_barrier(self, time: int) -> float:
        """Evaluates quality function based on classificator in the specified point.

        :return: Quality assessment.
        """
        raise NotImplementedError
