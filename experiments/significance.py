import os
from collections.abc import Sequence
from pathlib import Path

import numpy

from CPDShell.Core.algorithms.classification_algorithm import ClassificationAlgorithm
from CPDShell.Core.algorithms.ClassificationBasedCPD.test_statistics.threshold_overcome import ThresholdOvercome
from CPDShell.Core.scrubber.abstract_scrubber import Scrubber
from CPDShell.labeled_data import LabeledCPData
from CPDShell.shell import CPDShell
from experiments.statistics_counting import StatisticsCounting


class ThresholdCalculation:
    def __init__(
        self,
        knn_algorithm: ClassificationAlgorithm,
        scrubber: Scrubber,
        significance_level: float = 0.05,
        with_localization: bool = False,
        delta: float = 0.005,
    ) -> None:
        self.__knn_algorithm = knn_algorithm
        self.__scubber_class = scrubber
        self.__significance_level = significance_level
        self.__with_localization = with_localization
        self.__delta = delta

    def calculate_threshold(self, start_threshold: float, datasets_dir: Path) -> float:
        """
        :param datasets_dir: Path where datasets (without change point) are stored.
        """
        dataset = ThresholdCalculation.get_all_data(datasets_dir)
        cur_threshold = start_threshold
        cur_sig_level = self.__calculate_significance_level(cur_threshold, dataset)
        cur_difference = 1.0

        while abs(cur_sig_level - self.__significance_level) > self.__delta:
            print(cur_threshold)
            if cur_sig_level > self.__significance_level:
                cur_threshold = cur_threshold + cur_difference
                cur_sig_level = self.__calculate_significance_level(cur_threshold, dataset)

                if cur_sig_level < self.__significance_level + self.__delta:
                    cur_difference /= 2.0
            else:
                cur_threshold = cur_threshold - cur_difference
                cur_sig_level = self.__calculate_significance_level(cur_threshold, dataset)

                if cur_sig_level > self.__significance_level - self.__delta:
                    cur_difference /= 2.0

        return cur_threshold

    def __calculate_significance_level(self, threshold: float, dataset: list[Sequence[float | numpy.float64]]) -> float:
        change_points_count = 0
        overall_count = 0
        self.__knn_algorithm.test_statistic = ThresholdOvercome(threshold)

        for data in dataset:
            shell = CPDShell(data, cpd_algorithm=self.__knn_algorithm, scrubber=self.__scubber_class)

            result_container = shell.run_cpd()

            if self.__with_localization:
                change_points_count += len(result_container.result)
                overall_count += len(result_container.data)
            else:
                change_points_count += len(result_container.result) > 0
                overall_count += 1

        return change_points_count / overall_count

    def calculate_threshold_on_prepared_statistics(self, start_threshold: float, window_size: int, data_size: int, data_path: Path) -> float:
        cur_threshold = start_threshold
        cur_sig_level = self.__calculate_significance_level_on_prepared_statistics(cur_threshold, window_size, data_path)
        cur_difference = 1.0

        while abs(cur_sig_level - self.__significance_level) > self.__delta:
            print(cur_threshold)
            if cur_sig_level > self.__significance_level:
                cur_threshold = cur_threshold + cur_difference
                cur_sig_level = self.__calculate_significance_level_on_prepared_statistics(cur_threshold, window_size, data_path)

                if cur_sig_level < self.__significance_level + self.__delta:
                    cur_difference /= 2.0
            else:
                cur_threshold = cur_threshold - cur_difference
                cur_sig_level = self.__calculate_significance_level_on_prepared_statistics(cur_threshold, window_size, data_path)

                if cur_sig_level > self.__significance_level - self.__delta:
                    cur_difference /= 2.0

        return cur_threshold

    def __calculate_significance_level_on_prepared_statistics(threshold: float, window_size: int, data_path: Path) -> float:
        change_points = StatisticsCounting.get_change_points(data_path, ThresholdOvercome(threshold), window_size)
        change_points_count = len(change_points)
        overall_count = len()

        return change_points_count / overall_count

    @staticmethod
    def get_all_data(dataset_dir: Path) -> list[Sequence[float | numpy.float64]]:
        samples = ThresholdCalculation.get_all_sample_dirs(dataset_dir)
        print(samples)
        dataset = [LabeledCPData.read_generated_datasets(p)["normal"].raw_data for p in samples]

        return dataset

    @staticmethod
    def get_all_sample_dirs(dataset_dir: Path) -> list[Path]:
        root_content = os.listdir(dataset_dir)
        sample_paths = []

        for file in root_content:
            if not os.path.isdir(dataset_dir / file):
                continue

            if file.startswith("sample"):
                sample_paths.append(dataset_dir / file)
            else:
                sample_paths.extend(ThresholdCalculation.get_all_sample_dirs(dataset_dir / file))

        return sample_paths
