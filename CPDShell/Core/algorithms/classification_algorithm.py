"""
Module for implementation of CPD algorithm based on classification.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Iterable

import numpy as np

from CPDShell.Core.algorithms.abstract_algorithm import Algorithm
from CPDShell.Core.algorithms.ClassificationBasedCPD.abstracts.iclassifier import Classifier
from CPDShell.Core.algorithms.ClassificationBasedCPD.abstracts.istatistic_test import StatisticTest


class ClassificationAlgorithm(Algorithm):
    """
    The class implementing change point detection algorithm based on classification.
    """

    def __init__(
        self,
        classifier: Classifier,
        test_statistic: StatisticTest,
        shift_coeff: float
    ) -> None:
        """
        Initializes a new instance of classification based change point detection algorithm.

        :param classifier: Classifier for sample classification.
        :param test_statistic: Criterion to separate change points from other points in sample.
        :param shift_coeff: Coefficient for evaluating indent from window borders.
        The shift is calculated by multiplying the given coefficient by the size of window.
        """
        self.__classifiser = classifier
        self.__test_statistic = test_statistic
        self.__shift_coeff = shift_coeff

        self.__change_points: list[int] = []
        self.__change_points_count = 0

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
        sample = list(window)
        sample_size = len(sample)
        if sample_size == 0:
            return

        self.__classifiser.classify(window)

        # Examining each point.
        # Boundaries are always change points.
        first_point = int(sample_size * self.__shift_coeff)
        last_point = int(sample_size * (1 - self.__shift_coeff))
        assessments = []

        for time in range(first_point, last_point):
            quality = self.__classifiser.assess_in_point(time)
            assessments.append(quality)

        change_points = self.__test_statistic.get_change_points(assessments)

        # Shifting change points coordinates according to their place in window.
        self.__change_points = list(map(lambda x: x + first_point, change_points))
        self.__change_points_count = len(change_points)
