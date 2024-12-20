import os
from collections.abc import Sequence
from pathlib import Path

import numpy

from CPDShell.Core.algorithms.knn_algorithm import KNNAlgorithm
from CPDShell.Core.algorithms.classification_algorithm import ClassificationAlgorithm
from CPDShell.Core.algorithms.ClassificationBasedCPD.test_statistics.threshold_overcome import ThresholdOvercome
from CPDShell.Core.scrubber.abstract_scrubber import Scrubber
from CPDShell.labeled_data import LabeledCPData
from CPDShell.shell import CPDShell
from CPDShell.worker.utils import Utils
from experiments.rates import Rates


class ThresholdCalculation:
    @staticmethod
    def calculate_threshold(significance_level: float, threshold: float, data_size: int, dataset_path: Path, interval_length: int, delta: float) -> float:
        dataset = Utils.get_all_sample_dirs(dataset_path)

        cur_threshold = threshold
        cur_sig_level = ThresholdCalculation.__calculate_significance_level(cur_threshold, data_size, dataset, interval_length)
        cur_difference = 1.0

        while abs(cur_sig_level - significance_level) > delta:
            print(cur_threshold)
            if cur_sig_level > significance_level:
                cur_threshold = cur_threshold + cur_difference
                cur_sig_level = ThresholdCalculation.__calculate_significance_level(cur_threshold, data_size, dataset, interval_length)

                if cur_sig_level < significance_level + delta:
                    cur_difference /= 2.0
            else:
                cur_threshold = cur_threshold - cur_difference
                cur_sig_level = ThresholdCalculation.__calculate_significance_level(cur_threshold, data_size, dataset, interval_length)

                if cur_sig_level > significance_level - delta:
                    cur_difference /= 2.0

        return cur_threshold

    @staticmethod
    def __calculate_significance_level(threshold: float, data_size: int, dataset_path: list[tuple[Path, str]], interval_length: int) -> float:
        fpr_sum = 0
        overall_count = len(dataset_path)
        test_statistic = ThresholdOvercome(threshold)

        for data_path in dataset_path:
            fpr_sum += Rates.false_positive_rate(-1, data_path[0], test_statistic, data_size, window_size, interval_length)
            raise NotImplementedError
            fpr_sum += Rates.false_positive_rate(-1, data_path[0], test_statistic, data_size, indent_factor, window_size, interval_length)

        return fpr_sum / overall_count
