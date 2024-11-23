"""
Module for Classification CPD algorithm's classifier abstract base class.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np


class Classifier(ABC):
    """Classifier's abstract base class."""

    @abstractmethod
    def classify(self, window: Iterable[float | np.float64]) -> None:
        """Applies classificator to the given sample.

        :param window: part of global data for finding change points.
        """
        raise NotImplementedError

    @abstractmethod
    def assess_barrier(self, time: int) -> float:
        """Evaluates quality function based on classificator in the specified point.

        :param time: Index of point in the given sample to calculate quality.
        :return: Partitioning quality assessment.
        """
        raise NotImplementedError
