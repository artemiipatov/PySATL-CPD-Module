from pathlib import Path
import logging

from CPDShell.Core.algorithms.classification_algorithm import ClassificationAlgorithm
from CPDShell.Core.algorithms.ClassificationBasedCPD.test_statistics.threshold_overcome import ThresholdOvercome
from CPDShell.Core.algorithms.knn_algorithm import KNNAlgorithm
from CPDShell.Core.scrubber.linear_scrubber import LinearScrubber
from CPDShell.shell import CPContainer
from CPDShell.Worker.Common.statistics_calculation import StatisticsCalculation
from CPDShell.Worker.Common.utils import Utils
from CPDShell.Worker.worker import Worker


class CPDBenchmarkWorker(Worker):
    def __init__(self, expected_change_points: list[int], interval_length: int, logger: logging.Logger) -> None:
        self.__expected_change_points = expected_change_points
        self.__interval_length = interval_length

        self.__average_time: float = 0.0
        self.__power: float = 0.0
        self.__logger = logger

    def run(
        self,
        scrubber: LinearScrubber,
        cpd_algorithm: ClassificationAlgorithm | KNNAlgorithm,
        dataset_path: Path | None,
        results_path: Path,
    ) -> None:
        """Function for finding change points in window

        :param window: part of global data for finding change points
        :return: the number of change points in the window
        """
        # TODO: If dataset_path is not None ...
        assert dataset_path is not None, "Dataset path should not be None"

        results: list[CPContainer] = StatisticsCalculation.calculate_statistics(cpd_algorithm, scrubber, dataset_path, results_path)

        power = 0.0
        # Palliative. TODO: generalize
        for result in results:
            power += sum(
                map(
                    lambda x: any(
                        cp - self.__interval_length <= x <= cp + self.__interval_length for cp in result.result
                    ),
                    self.__expected_change_points,
                )
            ) / len(self.__expected_change_points)

        self.__power = power / len(results)
        self.__average_time = sum(result.time_sec for result in results) / len(results)
        self.__logger.info(f"Power: {self.__power}")
        self.__logger.info(f"Average time: {self.__average_time}")
        # Utils.print_all_change_points(results_path, ThresholdOvercome(self.__threshold), self.__interval_length)
