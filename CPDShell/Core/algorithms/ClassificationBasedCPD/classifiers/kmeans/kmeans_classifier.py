"""
Module for implementation of kmeans classifier for cpd.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Iterable

import numpy as np
from sklearn.cluster import KMeans

from CPDShell.Core.algorithms.ClassificationBasedCPD.abstracts.iclassifier import Classifier


class KMeansAlgorithm(Classifier):
    """
    The class implementing k-means classifier for cpd.
    """

    def __init__(
        self,
    ) -> None:
        """
        Initializes a new instance of k-means classifier for cpd.
        """
        self.__model: KMeans | None = None
        self.__window: list[float | np.float64] | None = None

    def classify(self, window: Iterable[float | np.float64]) -> None:
        """Applies classificator to the given sample.

        :param window: part of global data for finding change points
        """
        self.__window = list(window)
        k_means = KMeans(n_clusters=2)
        window_reshaped = np.array(self.__window).reshape(-1, 1)
        self.__model = k_means.fit(window_reshaped)

    def assess_barrier(self, time: int) -> float:
        """
        Calaulates quality function in specified point.

        :param time: index of point in the given sample to calculate statistics relative to it.
        """
        window_size = len(self.__window)
        length = min(window_size - time, time)
        start = max(0, time - length)
        end = min(window_size, time + length)
        labels = self.__model.labels_
        left = sum(labels[start:time])
        right = sum(labels[time:end])

        return abs(right - left) / length
