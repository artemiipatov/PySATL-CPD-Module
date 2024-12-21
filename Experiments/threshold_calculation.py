from pathlib import Path

from CPDShell.Core.algorithms.knn_algorithm import KNNAlgorithm
from CPDShell.Core.algorithms.classification_algorithm import ClassificationAlgorithm
from CPDShell.Core.scrubber.abstract_scrubber import Scrubber
from CPDShell.Worker.threshold_calculation_worker import ThresholdCalculationWorker
import Experiments.generator as Gen


class ThresholdCalculation:
    def __init__(
        self,
        cpd_algorithm: ClassificationAlgorithm | KNNAlgorithm,
        scrubber: Scrubber,
        interval_length: int,
        significance_level: float = 0.03,
        delta: float = 0.005,
    ) -> None:
        self.__cpd_algorithm = cpd_algorithm
        self.__scrubber = scrubber
        self.__interval_length = interval_length
        self.__significance_level = significance_level
        self.__delta = delta

    def calculate_threshold(self, distribution: Gen.Distribution, sample_count: int, dataset_path: Path | None, results_path: Path) -> float:
        """
        :param dataset_path: Path where datasets (without change point) should be saved.
        If dataset_path is None they are not supposed to be generated,
        that proposes that statistics is already calculated, and stored in results_path.
        """
        if dataset_path is not None:
            Gen.DistributionGenerator.generate([[distribution]], sample_count, dataset_path)
            worker = ThresholdCalculationWorker(self.__significance_level, self.__delta, distribution.length, self.__interval_length)

        threshold = worker.run(self.__scrubber, None, self.__cpd_algorithm, dataset_path, results_path)
        return threshold
