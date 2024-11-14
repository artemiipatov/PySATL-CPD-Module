"""
Module for implementation of random forest classifier for cpd.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

import typing as tp
from collections.abc import Iterable
from math import sqrt

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

    @property
    def window(self) -> list[float | np.float64] | None:
        return self.__window
    
    @window.setter
    def window(self, val: Iterable[float | np.float64]) -> None:
        self.__window = list(val)

    def classify_barrier(self, time: int) -> float:
        """
        Calaulates quality function in specified point.

        :param time: index of point in the given sample to calculate statistics relative to it.
        """
        train_sample_X = [[x] for i, x in enumerate(self.window) if i % 2 == 0]
        train_sample_Y = [int(i > time) for i in range(0, self.__window, 2)]
        test_sample = [[x] for i, x in enumerate(self.window) if i % 2 != 0]

        self.__model = RandomForestClassifier()
        self.__model.fit(train_sample_X, train_sample_Y)
        prediction = self.__model.predict(test_sample)
        right = prediction.sum()
        left = len(prediction) - right
        
        return 2 * min(right, left) / len(prediction)
