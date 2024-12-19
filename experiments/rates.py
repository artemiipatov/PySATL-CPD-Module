from pathlib import Path

from CPDShell.Core.algorithms.ClassificationBasedCPD.abstracts.istatistic_test import TestStatistic
from experiments.statistics_calculation import StatisticsCalculation


class Rates:
    @staticmethod
    def false_negative_rate(change_point_i: int, statistics_dir: Path, test_statistic: TestStatistic, dataset_size: int, indent_factor: float, window_size: int, delta: int):
        if change_point_i < 0:
            return 0

        change_points = StatisticsCalculation.get_change_points(statistics_dir, test_statistic, window_size)
        change_point_i = change_point_i - window_size * indent_factor

        true_positives = list(filter(lambda x: change_point_i - delta <= x <= change_point_i + delta, change_points))
        false_negatives = 0 if true_positives else 1

        return false_negatives
    
    @staticmethod
    def true_positive_rate(change_point_i: int, statistics_dir: Path, test_statistic: TestStatistic, dataset_size: int, indent_factor: float, window_size: int, delta: int):
        return 1 - Rates.false_negative_rate(change_point_i, statistics_dir, test_statistic, dataset_size, indent_factor, window_size, delta)
    
    @staticmethod
    def false_positive_rate(change_point_i: int, statistics_dir: Path, test_statistic: TestStatistic, dataset_size: int, indent_factor: float, window_size: int, delta: int):
        change_points = StatisticsCalculation.get_change_points(statistics_dir, test_statistic, window_size)
        change_point_i = change_point_i - int(window_size * indent_factor) if change_point_i >= 0 else change_point_i
        overall_count = (dataset_size - 2 * int(window_size * indent_factor)) // (2 * delta)

        start = (change_point_i - delta) % (2 * delta) if change_point_i >= 0 else 0
        predicted_positives = 1 if change_points[:start] else 0
        while start <= dataset_size:
            predicted_positives += 1 if list(filter(lambda x: start <= x < start + 2 * delta, change_points)) else 0
            start += 2 * delta + 1 # TODO: Think over 1 addition

        true_positives = 1 if change_point_i >= 0 and list(filter(lambda x: change_point_i - delta <= x <= change_point_i + delta, change_points)) else 0
        false_positives = predicted_positives - true_positives
        negatives = overall_count - 1

        return false_positives / negatives

    @staticmethod
    def true_negative_rate(change_point_i: int, statistics_dir: Path, test_statistic: TestStatistic, dataset_size: int, indent_factor: float, window_size: int, delta: int):
        return 1 - Rates.false_positive_rate(change_point_i, statistics_dir, test_statistic, dataset_size, indent_factor, window_size, delta)
