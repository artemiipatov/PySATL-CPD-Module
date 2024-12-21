from pathlib import Path

from CPDShell.Core.algorithms.classification_algorithm import ClassificationAlgorithm
from CPDShell.Core.algorithms.ClassificationBasedCPD.test_statistics.threshold_overcome import ThresholdOvercome
from CPDShell.Core.algorithms.knn_algorithm import KNNAlgorithm
from CPDShell.Core.scrubber.linear_scrubber import LinearScrubber
from CPDShell.Core.scrubber_scenario import ScrubberScenario
from CPDShell.Worker.Common.statistics_calculation import StatisticsCalculation
from CPDShell.Worker.Common.utils import Utils
from CPDShell.Worker.worker import Worker


class CPDWorker(Worker):
    def __init__(self, interval_length: int, threshold: float) -> None:
        self.__interval_length = interval_length
        self.__threshold = threshold

    def run(
        self,
        scrubber: LinearScrubber,
        scenario: ScrubberScenario | None,
        cpd_algorithm: ClassificationAlgorithm | KNNAlgorithm,
        dataset_path: Path | None,
        results_path: Path,
    ) -> None:
        """Function for finding change points in window

        :param window: part of global data for finding change points
        :return: the number of change points in the window
        """
        if dataset_path is not None:
            statistics_calculation = StatisticsCalculation(cpd_algorithm, scrubber)
            statistics_calculation.calculate_statistics(dataset_path, results_path)

        Utils.print_all_change_points(results_path, ThresholdOvercome(self.__threshold), self.__interval_length)
