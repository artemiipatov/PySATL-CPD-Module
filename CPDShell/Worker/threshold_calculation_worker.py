from pathlib import Path

from CPDShell.Core.algorithms.classification_algorithm import ClassificationAlgorithm
from CPDShell.Core.algorithms.knn_algorithm import KNNAlgorithm
from CPDShell.Core.scrubber.linear_scrubber import LinearScrubber
from CPDShell.Core.scrubber_scenario import ScrubberScenario
from CPDShell.Worker.Common.statistics_calculation import StatisticsCalculation
from CPDShell.Worker.Common.threshold_calculation import ThresholdCalculation
from CPDShell.Worker.worker import Worker


class ThresholdCalculationWorker(Worker):
    def __init__(self, significance_level: float, sl_delta: float, sample_length: int, interval_length: int) -> None:
        self.__significance_level = significance_level
        self.__sl_delta = sl_delta
        self.__sample_length = sample_length
        self.__interval_length = interval_length
        self.__threshold = 0.0

    @property
    def threshold(self) -> float:
        return self.__threshold

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

        threshold = ThresholdCalculation.calculate_threshold(
            self.__significance_level,
            1.0,
            self.__sample_length,
            scrubber.window_length,
            results_path,
            self.__interval_length,
            self.__sl_delta,
        )

        self.__threshold = threshold

        print(f"Optimal threshold for significance level {self.__significance_level}: {threshold}")
