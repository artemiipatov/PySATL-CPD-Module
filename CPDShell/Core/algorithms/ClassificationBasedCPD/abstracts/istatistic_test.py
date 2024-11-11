from abc import ABC, abstractmethod


class StatisticTest(ABC):
    """Abstract class for change point detection algorithms"""

    @abstractmethod
    def get_change_points(self, classifier_assesments: list[float]) -> list[int]:
        """Separates change points from other points in sample based on some criterion.

        :param classifier_assesments: List of quality assessments evaluated in each point of the sample.
        :return: Change points in the current window.
        """
        raise NotImplementedError
