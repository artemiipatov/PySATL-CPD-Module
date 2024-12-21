from abc import ABC, abstractmethod
from pathlib import Path

from CPDShell.Core.algorithms.classification_algorithm import ClassificationAlgorithm
from CPDShell.Core.algorithms.knn_algorithm import KNNAlgorithm
from CPDShell.Core.scrubber.linear_scrubber import LinearScrubber
from CPDShell.Core.scrubber_scenario import ScrubberScenario


class Worker(ABC):
    """Abstract class for worker"""

    @abstractmethod
    def run(
        self,
        scrubber: LinearScrubber,
        scenario: ScrubberScenario | None,
        cpd_algorithm: ClassificationAlgorithm | KNNAlgorithm,
        dataset_path: Path,
        results_path: Path,
    ) -> None:
        """Function for finding change points in window

        :param window: part of global data for finding change points
        :return: the number of change points in the window
        """
        raise NotImplementedError
