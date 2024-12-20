import os
from collections.abc import Sequence
from pathlib import Path
from CPDShell.Core.algorithms.ClassificationBasedCPD.abstracts.istatistic_test import TestStatistic
from CPDShell.worker.statistics_calculation import StatisticsCalculation
from CPDShell.labeled_data import LabeledCPData

import numpy


class Utils:
    @staticmethod
    def read_data(data_path: Path) -> list[numpy.float64]:
        with open(data_path) as infile:
            data = list(map(numpy.float64, infile.readlines()))

        return data

    @staticmethod
    def get_change_points(data: list[float | numpy.float64], test_statistic: TestStatistic, window_size: int) -> list[int]:
        change_points = []

        for start in range(0, len(data), window_size):
            change_points += list(map(lambda x: x + start, test_statistic.get_change_points(data[start : start + window_size])))

        return change_points

    @staticmethod
    def print_all_change_points(statistics_dir: Path, test_statistic: TestStatistic, window_size: int) -> None:
        stats_dirs = Utils.get_all_stats_dirs(statistics_dir)
        for stats_path in stats_dirs:
            change_points = Utils.get_change_points(stats_path, test_statistic, window_size)
            print(stats_path)
            print(change_points)

    @staticmethod
    def get_all_stats_dirs(statistics_dir: Path) -> list[Path]:
        root_content = os.listdir(statistics_dir)
        sample_paths = []

        for name in root_content:
            cur_path = statistics_dir / name

            if name == "stats":
                sample_paths.append(cur_path)
                continue

            if os.path.isdir(cur_path):
                sample_paths.extend(Utils.get_all_stats_dirs(cur_path))

        return sample_paths

    @staticmethod
    def read_all_data_from_dir(dataset_dir: Path) -> list[Sequence[float | numpy.float64]]:
        samples = Utils.get_all_sample_dirs(dataset_dir)
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
                sample_paths.extend(Utils.get_all_sample_dirs(cur_path))

        return sample_paths