"""
Module for implementation of classifier based on nearest neighbours for cpd.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

import typing as tp
from collections.abc import Iterable
from math import sqrt

import numpy as np

from CPDShell.Core.algorithms.ClassificationBasedCPD.abstracts.iclassifier import Classifier
from CPDShell.Core.algorithms.ClassificationBasedCPD.classifiers.knn.knn_graph import KNNGraph


class KNNAlgorithm(Classifier):
    """
    The class implementing classifier based on nearest neighbours.
    """

    def __init__(
        self,
        metric: tp.Callable[[float, float], float] | tp.Callable[[np.float64, np.float64], float],
        k=7,
        delta: float = 1e-12,
    ) -> None:
        """
        Initializes a new instance of KNN classifier for cpd.

        :param metric: function for calculating distance between points in time series.
        :param k: number of neighbours in graph relative to each point.
        :param delta: delta for comparing floats.
        """
        self.__k = k
        self.__metric = metric
        self.__delta = delta

        self.__window: list[float | np.float64] = None
        self.__knn_graph: KNNGraph | None = None

    def classify(self, window: Iterable[float | np.float64]) -> None:
        """Applies classificator to the given sample.

        :param window: part of global data for finding change points.
        """
        self.__window = list(window)
        self.__knn_graph = KNNGraph(window, self.__metric, self.__k, self.__delta)
        self.__knn_graph.build()

    def assess_barrier(self, time: int) -> float:
        """
        Calaulates quality function in specified point.

        :param time: index of point in the given sample to calculate statistics relative to it.
        """
        window_size = len(self.__window)

        assert self.__knn_graph is not None, "Graph should not be None."

        k = self.__k
        n = window_size
        n_1 = time
        n_2 = n - time

        if n <= k:
            # Unable to analyze sample due to its size.
            # Returns negative number that will be less than statistics in this case,
            # but big enough not to spoil visualization.
            return -k

        h = 4 * (n_1 - 1) * (n_2 - 1) / ((n - 2) * (n - 3))

        sum_1 = (1 / n) * sum(
            self.__knn_graph.check_for_neighbourhood(j, i)
            for i in range(window_size)
            for j in self.__knn_graph.get_neighbours(i)
        )

        sum_2 = (1 / n) * (
            2
            * sum(
                self.__knn_graph.check_for_neighbourhood(m, i)
                for j in range(window_size)
                for i in self.__knn_graph.get_neighbours(j)
                for m in range(j + 1, window_size)
            )
            + sum(len(self.__knn_graph.get_neighbours(i)) for i in range(window_size))
        )

        expectation = 4 * k * n_1 * n_2 / (n - 1)
        variance = (expectation / k) * (h * (sum_1 + k - (2 * k**2 / (n - 1))) + (1 - h) * (sum_2 - k**2))
        deviation = sqrt(variance)

        permutation: np.array = np.arange(window_size)
        random_variable_value = self.__calculate_random_variable(permutation, time, window_size)

        if deviation == 0:
            # if deviation is zero, it likely means that time is 1. This implies that h is 0 and sum_2 = k**2.
            # In this case we can for sure say that there is no change-point.
            # Expectation in this case is equal to 4 * k, and random variable less or equal to 2.
            # Thus returning negative difference of them will be enough not to increase false positive.
            return -(random_variable_value - expectation)

        statistics = -(random_variable_value - expectation) / deviation

        return statistics

    def __calculate_random_variable(self, permutation: np.array, t: int, window_size: int) -> int:
        """
        Calculates a random variable from a permutation and a fixed point.

        :param permutation: random permutation of observations.
        :param t: fixed point that splits the permutation.
        :return: value of the random variable.
        """

        def b(i: int, j: int) -> bool:
            pi = permutation[i]
            pj = permutation[j]
            return (pi <= t < pj) or (pj <= t < pi)

        s = sum(
            (self.__knn_graph.check_for_neighbourhood(i, j) + self.__knn_graph.check_for_neighbourhood(j, i)) * b(i, j)
            for i in range(window_size)
            for j in range(window_size)
        )

        return s
