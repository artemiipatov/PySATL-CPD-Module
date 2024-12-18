import os
from collections.abc import Sequence
from pathlib import Path

import numpy

from CPDShell.Core.algorithms.classification_algorithm import ClassificationAlgorithm
from CPDShell.Core.algorithms.ClassificationBasedCPD.test_statistics.threshold_overcome import ThresholdOvercome
from CPDShell.Core.scrubber.abstract_scrubber import Scrubber
from CPDShell.labeled_data import LabeledCPData
from CPDShell.shell import CPDShell
from experiments.statistics_calculation import StatisticsCalculation
from experiments.rates import Rates


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

    def calculate_threshold_on_prepared_statistics(self, threshold: float, window_size: int, data_size: int, dataset_path: Path, indent_factor: float, delta: int) -> float:
        dataset = ThresholdCalculation.get_all_sample_dirs(dataset_path)

        cur_threshold = threshold
        cur_sig_level = ThresholdCalculation.__calculate_significance_level_on_prepared_statistics(cur_threshold, window_size, data_size, dataset, indent_factor, delta)
        cur_difference = 1.0

        while abs(cur_sig_level - self.__significance_level) > self.__delta:
            print(cur_threshold)
            if cur_sig_level > self.__significance_level:
                cur_threshold = cur_threshold + cur_difference
                cur_sig_level = ThresholdCalculation.__calculate_significance_level_on_prepared_statistics(cur_threshold, window_size, data_size, dataset, indent_factor, delta)

                if cur_sig_level < self.__significance_level + self.__delta:
                    cur_difference /= 2.0
            else:
                cur_threshold = cur_threshold - cur_difference
                cur_sig_level = ThresholdCalculation.__calculate_significance_level_on_prepared_statistics(cur_threshold, window_size, data_size, dataset, indent_factor, delta)

                if cur_sig_level > self.__significance_level - self.__delta:
                    cur_difference /= 2.0

        return cur_threshold

    @staticmethod
    def __calculate_significance_level_on_prepared_statistics(threshold: float, window_size: int, data_size: int, dataset_path: list[tuple[Path, str]], indent_factor: float, delta: int) -> float:
        fpr_sum = 0
        overall_count = len(dataset_path)
        test_statistic = ThresholdOvercome(threshold)

        for data_path in dataset_path:
            fpr_sum += Rates.false_positive_rate(-1, data_path[0], test_statistic, data_size, indent_factor, window_size, delta)

        return fpr_sum / overall_count

    @staticmethod
    def get_all_data(dataset_dir: Path) -> list[Sequence[float | numpy.float64]]:
        samples = ThresholdCalculation.get_all_sample_dirs(dataset_dir)
        dataset = [LabeledCPData.read_generated_datasets(p[0])[p[1]].raw_data for p in samples]

        return dataset

    @staticmethod
    def get_all_sample_dirs(dataset_dir: Path) -> list[tuple[Path, str]]:
        root_content = os.listdir(dataset_dir)
        sample_paths = []

        for name in root_content:
            cur_path = dataset_dir / name

            if not os.path.isdir(cur_path):
                continue

            if name.startswith("sample"):
                distr_name = os.listdir(cur_path)[0]
                sample_paths.append((cur_path, distr_name))
            else:
                sample_paths.extend(StatisticsCalculation.get_all_sample_dirs(cur_path))

        return sample_paths
