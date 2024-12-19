import os
from collections.abc import Sequence
from pathlib import Path

import numpy

from CPDShell.Core.algorithms.classification_algorithm import ClassificationAlgorithm
from CPDShell.Core.algorithms.knn_algorithm import KNNAlgorithm
from CPDShell.Core.algorithms.ClassificationBasedCPD.abstracts.istatistic_test import TestStatistic
from CPDShell.Core.scrubber.abstract_scrubber import Scrubber
from CPDShell.labeled_data import LabeledCPData
from CPDShell.shell import CPDShell


class StatisticsCalculation:
    def __init__(
        self,
        cpd_algorithm: ClassificationAlgorithm | KNNAlgorithm,
        scrubber: Scrubber,
    ) -> None:
        self.__cpd_algorithm = cpd_algorithm
        self.__scubber_class = scrubber

    def calculate_statistics(self, datasets_dir: Path, dest_dir: Path):
        """
        :param datasets_dir: Path where datasets are stored.
        """
        samples_dirs = StatisticsCalculation.get_all_sample_dirs(datasets_dir)

        for sample_dir in samples_dirs:
            data = LabeledCPData.read_generated_datasets(sample_dir[0])[sample_dir[1]].raw_data
            shell = CPDShell(data, cpd_algorithm=self.__cpd_algorithm, scrubber=self.__scubber_class)
            shell.run_cpd()

            stats = self.__cpd_algorithm.statistics_list
            dest_path = dest_dir / sample_dir[0].parts[sample_dir[0].parts.index(sample_dir[1]) - 1] / sample_dir[1] / sample_dir[0].name
            os.makedirs(dest_path, exist_ok=True)

            with open(dest_path / "stats", "w+") as outfile:
                for stat in stats:
                    outfile.write(str(stat) + "\n")

            self.__cpd_algorithm.statistics_list = []
            print(sample_dir[0])

    @staticmethod
    def get_change_points(dataset_path: Path, test_statistic: TestStatistic, window_size: int):
        with open(dataset_path / "stats") as infile:
            data = list(map(numpy.float64, infile.readlines()))

        change_points = []

        for start in range(0, len(data), window_size):
            change_points += list(map(lambda x: x + start, test_statistic.get_change_points(data[start : start + window_size])))

        return change_points

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
