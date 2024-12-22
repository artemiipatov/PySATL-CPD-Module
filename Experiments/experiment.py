import typing as tp
from pathlib import Path
from shutil import rmtree

import logging
import yaml

import Experiments.generator as Gen
from CPDShell.Core.algorithms.classification_algorithm import ClassificationAlgorithm
from CPDShell.Core.algorithms.ClassificationBasedCPD.test_statistics.threshold_overcome import ThresholdOvercome
from CPDShell.Core.algorithms.knn_algorithm import KNNAlgorithm
from CPDShell.Core.scrubber.linear_scrubber import LinearScrubber
from CPDShell.Worker.threshold_calculation_worker import ThresholdCalculationWorker
from CPDShell.Worker.cpd_worker import CPDBenchmarkWorker


SAMPLE_COUNT_FOR_THRESHOLD_CALC = 100
WITHOUT_CP_SAMPLE_LENGTH = 200

class Experiments:
    def __init__(
        self,
        cpd_algorithm: ClassificationAlgorithm | KNNAlgorithm,
        scrubber: LinearScrubber,
        logger: logging.Logger
    ) -> None:
        """
        :param config_path: path to yaml file with distributions configuration.
        :param dataset_path: path to directory where generated datasets should be saved.
        :param dataset_path: path to directory where statistics should be saved.
        """
        self.__cpd_algorithm = cpd_algorithm
        self.__scrubber = scrubber
        self.__logger = logger

    def run(self,
            significance_level: float,
            sample_count: int,
            interval_length: int,
            expected_change_points: list[int],
            config_path: Path,
            dataset_path: Path,
            results_path: Path,
            delta: float = 0.005
        ) -> None:
        distributions = Gen.DistributionGenerator.generate_by_config(config_path, dataset_path, sample_count)

        for i in range(len(distributions)):
            distr_comp = distributions[i]

            self.__logger.info(f"Distribution {distr_comp}")
            # Generating the dataset without change points.
            without_cp_path = dataset_path / "without_cp"
            Path(without_cp_path).mkdir(parents=True, exist_ok=True)
            without_cp_distr = Gen.Distribution(distr_comp[0].type, distr_comp[0].parameters, WITHOUT_CP_SAMPLE_LENGTH)
            Gen.DistributionGenerator.generate([[without_cp_distr]], SAMPLE_COUNT_FOR_THRESHOLD_CALC, without_cp_path)

            # Calculating threshold on the dataset without change points and according to the given significance level.
            sample_length = sum(distr.length for distr in distr_comp)
            threshold_calculation = ThresholdCalculationWorker(significance_level, delta, sample_length, interval_length, self.__logger)
            threshold_calculation.run(self.__scrubber, self.__cpd_algorithm, without_cp_path, results_path)
            threshold = threshold_calculation.threshold

            # Removing the generated dataset without change points.
            rmtree(without_cp_path)

            # Run benchmark with calculated threshold.
            distr_path = dataset_path / Path(f"{i}-" + "-".join(map(lambda d: d.type.name, distr_comp)))
            cpd = CPDBenchmarkWorker(expected_change_points, interval_length, self.__logger)
            self.__cpd_algorithm.test_statistic = ThresholdOvercome(threshold)
            cpd.run(self.__scrubber, self.__cpd_algorithm, distr_path, results_path)
