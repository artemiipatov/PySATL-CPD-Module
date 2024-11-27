"""
Module for implementation of svm classifier for cpd.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
from sklearn.svm import SVC

from CPDShell.Core.algorithms.ClassificationBasedCPD.abstracts.iclassifier import Classifier


class SVMAlgorithm(Classifier):
    """
    The class implementing svm classifier for cpd.
    """

    def __init__(
        self,
    ) -> None:
        """
        Initializes a new instance of k-means classifier for cpd.
        """
        self.__model: SVC | None = None

    def train(
        self,
        sample: list[float | np.float64],
        barrier: int
    ) -> None:
        """Trains classifier on the given sample.

        :param sample: sample for training classifier.
        """
        classes = [0 if i <= barrier else 1 for i in range(len(sample))]
        self.__model = SVC()
        self.__model.fit(sample, classes)

    def predict(
        self,
        sample: list[float | np.float64]
    ) -> np.ndarray:
        """Applies classificator to the given sample.

        :param window: part of global data for finding change points.
        """
        return self.__model.predict(sample)
