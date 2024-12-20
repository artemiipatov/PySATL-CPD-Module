from abc import ABC, abstractmethod
from collections.abc import MutableSequence
from pathlib import Path

from CPDShell.Core.scrubber.abstract_scrubber import Scrubber
from CPDShell.Core.scrubber_scenario import ScrubberScenario
from CPDShell.Core.algorithms.knn_algorithm import KNNAlgorithm
from CPDShell.Core.algorithms.classification_algorithm import ClassificationAlgorithm
from CPDShell.Core.algorithms.ClassificationBasedCPD.test_statistics.threshold_overcome import ThresholdOvercome
from CPDShell.Core.scrubber.linear_scrubber import LinearScrubber
from CPDShell.labeled_data import LabeledCPData
from CPDShell.shell import CPDShell
from experiments.statistics_calculation import StatisticsCalculation
from experiments.rates import Rates

import numpy


class Worker(ABC):
    """Abstract class for worker"""

    @abstractmethod
    def run(
        self,
        scrubber: LinearScrubber,
        scenario: ScrubberScenario | None,
        cpd_algorithm: ClassificationAlgorithm | KNNAlgorithm,
        dataset_path: Path,
        results_path: Path
    ) -> None:
        """Function for finding change points in window

        :param window: part of global data for finding change points
        :return: the number of change points in the window
        """
        raise NotImplementedError
