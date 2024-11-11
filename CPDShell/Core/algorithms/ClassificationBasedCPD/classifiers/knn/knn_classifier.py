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

from CPDShell.Core.algorithms.ClassificationBasedCPD.classifiers.knn.knn_graph import KNNGraph
from CPDShell.Core.algorithms.ClassificationBasedCPD.abstracts.iclassifier import Classifier


class KNNAlgorithm(Classifier):
    """
    The class implementing classifier based on nearest neighbours.
    """

    def __init__(
        self,
        metric: tp.Callable[[float, float], float] | tp.Callable[[np.float64, np.float64], float],
        k=3,
        delta: float = 1e-12,
    ) -> None:
        """
        Initializes a new instance of KNN classifier for cpd.

        :param metric: function for calculating distance between points in time series.
        :param k: number of neighbours in graph relative to each point.
        """
        self.__k = k
        self.__metric = metric
        self.__delta = delta

<<<<<<< HEAD:CPDShell/Core/algorithms/knn_algorithm.py
        self.__change_points: list[int] = []
        self.__change_points_count = 0

        self.__knn_graph: knngraph.KNNGraph | None = None

        self.statistics_list: list[float | np.float64] = []
        self.global_time = 0

    def detect(self, window: Iterable[float | np.float64]) -> int:
        """Finds change points in window.

        :param window: part of global data for finding change points.
        :return: the number of change points in the window.
        """
        self.__process_data(False, window)
        return self.__change_points_count

    def localize(self, window: Iterable[float | np.float64]) -> list[int]:
        """Finds coordinates of change points (localizes them) in window.

        :param window: part of global data for finding change points.
        :return: list of window change points.
        """
        self.__process_data(window)
        return self.__change_points.copy()

    def __process_data(self, window: Iterable[float | np.float64]) -> None:
        """
        Processes a window of data to detect/localize all change points depending on working mode.

        :param window: part of global data for change points analysis.
        """
        sample = deque(window)
        sample_size = len(sample)
        if sample_size == 0:
            return

        # Preparing.
        self.__change_points: list[int] = []
        self.__change_points_count = 0
=======
        self.__window_size = 0
        self.__knn_graph: KNNGraph | None = None
>>>>>>> knn-cpd:CPDShell/Core/algorithms/ClassificationBasedCPD/classifiers/knn/knn_classifier.py

    def classify(self, window: Iterable[float | np.float64]) -> None:
        # Building graph.
        self.__knn_graph = KNNGraph(window, self.__metric, self.__k, self.__delta)
        self.__knn_graph.build()
        self.__window_size = len(list(window))

<<<<<<< HEAD:CPDShell/Core/algorithms/knn_algorithm.py
        # Examining each point.
        # Boundaries are always change points.
        first_point = int(len(window) * 0.25)
        last_point = int(len(window) * 0.75)

        for time in range(first_point, last_point):
            statistics = self.__calculate_statistics_in_point(time, len(window))
            self.statistics_list.append(statistics)
            self.global_time += 1
            # print(time, statistics)
            if self.__check_change_point(statistics):
                self.__change_points.append(time)
                self.__change_points_count += 1

    def __calculate_statistics_in_point(self, time: int, window_size: int) -> float:
=======
    def assess_in_point(self, time: int) -> float:
>>>>>>> knn-cpd:CPDShell/Core/algorithms/ClassificationBasedCPD/classifiers/knn/knn_classifier.py
        """
        Calaulates quality function in specified point.

        :param time: index of point in the given sample to calculate statistics relative to it.
        """
        window_size = self.__window_size

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
            self.__knn_graph.check_for_neighbourhood(i, j) * self.__knn_graph.check_for_neighbourhood(j, i)
            for i in range(window_size)
            for j in range(window_size)
        )

        sum_2 = (1 / n) * sum(
            self.__knn_graph.check_for_neighbourhood(j, i) * self.__knn_graph.check_for_neighbourhood(m, i)
            for i in range(window_size)
            for j in range(window_size)
            for m in range(window_size)
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
