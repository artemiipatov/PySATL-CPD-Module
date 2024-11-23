"""
Module for implementation of test statistic based on threshold overcome.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

from CPDShell.Core.algorithms.ClassificationBasedCPD.abstracts.istatistic_test import TestStatistic


class ThresholdOvercome(TestStatistic):
    """
    The class implementing test statistic based on threshold overcome.
    """

    def __init__(self, threshold: float) -> None:
        """
        Initializes a new instance of threshold overcome criterion.

        :param threshold: Threshold to overcome to detect the change point.
        """
        self.__threshold = threshold

    def get_change_points(self, classifier_assesments: list[float]) -> list[int]:
        """Separates change points from other points in sample based on some criterion.

        :param classifier_assesments: List of quality assessments evaluated in each point of the sample.
        :return: Change points in the current window.
        """
        return [i for i, v in enumerate(classifier_assesments) if v > self.__threshold]
