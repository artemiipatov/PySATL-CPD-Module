import os
from pathlib import Path

import numpy

from CPDShell.Core.algorithms.classification_algorithm import ClassificationAlgorithm
from CPDShell.Core.algorithms.knn_algorithm import KNNAlgorithm
from CPDShell.Core.algorithms.ClassificationBasedCPD.abstracts.istatistic_test import TestStatistic
from CPDShell.Core.scrubber.abstract_scrubber import Scrubber
from CPDShell.labeled_data import LabeledCPData
from CPDShell.shell import CPDShell
from CPDShell.worker.utils import Utils


class StatisticsCalculation:
    @staticmethod
    def calculate_statistics(cpd_algorithm: ClassificationAlgorithm | KNNAlgorithm, scrubber: Scrubber, datasets_dir: Path, dest_dir: Path):
        """
        :param datasets_dir: Path where datasets are stored.
        """
        samples_dirs = Utils.get_all_sample_dirs(datasets_dir)

        for sample_dir in samples_dirs:
            data = LabeledCPData.read_generated_datasets(sample_dir[0])[sample_dir[1]].raw_data
            shell = CPDShell(data, cpd_algorithm=cpd_algorithm, scrubber=scrubber)
            shell.run_cpd()

            stats = cpd_algorithm.statistics_list
            dest_path = dest_dir / sample_dir[0].parts[sample_dir[0].parts.index(sample_dir[1]) - 1] / sample_dir[1] / sample_dir[0].name
            os.makedirs(dest_path, exist_ok=True)

            with open(dest_path / "stats", "w+") as outfile:
                for stat in stats:
                    outfile.write(str(stat) + "\n")

            cpd_algorithm.statistics_list = []
            print(sample_dir[0])
