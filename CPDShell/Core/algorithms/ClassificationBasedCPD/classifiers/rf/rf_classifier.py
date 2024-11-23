"""
Module for implementation of random forest classifier for cpd.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Iterable

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from CPDShell.Core.algorithms.ClassificationBasedCPD.abstracts.iclassifier import Classifier


class RFClassifier(Classifier):
    """
    The class implementing random forest classifier for cpd.
    """

    def __init__(
        self,
    ) -> None:
        """
        Initializes a new instance of k-means classifier for cpd.
        """
        self.__model: RandomForestClassifier | None = None
        self.__window: list[float | np.float64] = None

    def classify(self, window: Iterable[float | np.float64]) -> None:
        """Applies classificator to the given sample.

        :param window: part of global data for finding change points.
        """
        self.__window = list(window)

    def assess_barrier(self, time: int) -> float:
        """
        Calaulates quality function in specified point.

        :param time: index of point in the given sample to calculate statistics relative to it.
        """
        train_sample_X = [[x] for i, x in enumerate(self.window) if i % 2 == 0]
        train_sample_Y = [int(i > time) for i in range(0, len(self.__window), 2)]
        test_sample = [[x] for i, x in enumerate(self.window) if i % 2 != 0]

        self.__model = RandomForestClassifier()
        self.__model.fit(train_sample_X, train_sample_Y)
        prediction = self.__model.predict(test_sample)
        right = prediction.sum()
        left = len(prediction) - right

        return 2 * min(right, left) / len(prediction)
