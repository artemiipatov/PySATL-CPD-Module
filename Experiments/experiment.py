import typing as tp
from pathlib import Path
from shutil import rmtree

import yaml

import Experiments.generator as Gen
from CPDShell.Core.algorithms.classification_algorithm import ClassificationAlgorithm
from CPDShell.Core.algorithms.knn_algorithm import KNNAlgorithm
from CPDShell.Core.scrubber.abstract_scrubber import Scrubber
from CPDShell.Worker.threshold_calculation_worker import ThresholdCalculationWorker
from CPDShell.Worker.cpd_worker import CPDWorker
from CPDShell.Worker.Common.rates import Rates


SAMPLE_COUNT_FOR_THRESHOLD_CALC = 200

class Experiments:
    def __init__(
        self,
        cpd_algorithm: ClassificationAlgorithm | KNNAlgorithm,
        scrubber: Scrubber,
        config_path: Path,
        dataset_path: Path,
        results_path: Path,
    ) -> None:
        """
        :param config_path: path to yaml file with distributions configuration.
        :param dataset_path: path to directory where generated datasets should be saved.
        :param dataset_path: path to directory where statistics should be saved.
        """
        self.__cpd_algorithm = cpd_algorithm
        self.__scrubber = scrubber

    def run(self,
            significance_level: float,
            sample_count: int,
            interval_length: int,
            config_path: Path,
            dataset_path: Path,
            results_path: Path,
            delta: float = 0.005
        ) -> None:
        distributions = Experiments.generate_by_config(config_path, dataset_path, sample_count)

        for distr_comp in distributions:
            # Generating the dataset without change points.
            without_cp_path = dataset_path / "without_cp"
            Path(without_cp_path).mkdir(parents=True, exist_ok=True)
            Gen.DistributionGenerator.generate([distr_comp[0]], SAMPLE_COUNT_FOR_THRESHOLD_CALC, without_cp_path)

            # Calculating threshold on the dataset without change points and according to the given significance level.
            sample_length = sum(distr.length for distr in distr_comp)
            threshold_calculation = ThresholdCalculationWorker(significance_level, delta, sample_length, interval_length)
            threshold_calculation.run(self.__scrubber, None, self.__cpd_algorithm, without_cp_path, results_path)
            threshold = threshold_calculation.threshold

            # Removing the generated dataset without change points.
            rmtree(without_cp_path)

            distr_path = dataset_path / "-".join(map(lambda d: d.type.name, distr_comp))
            cpd = CPDWorker(interval_length, threshold)
            cpd.run(self.__scrubber, None, self.__cpd_algorithm, distr_path, results_path)

    @staticmethod
    def generate_by_config(config_path: Path, dataset_path: Path, sample_count: int) -> list[Gen.DistributionComposition]:
        with open(config_path) as stream:
            loaded_config: list[dict[str, tp.Any]] = yaml.safe_load(stream)

        distributions: list[Gen.DistributionComposition] = []

        for distr_comp_config in loaded_config:
            distr_comp: Gen.DistributionComposition = [
                Gen.Distribution(distr_config["type"], distr_config["parameters"], distr_config["length"])
                for distr_config in distr_comp_config["distributions"]
            ]
            distributions.append(distr_comp)

        Gen.DistributionGenerator.generate(distributions, sample_count, dataset_path)

        return distributions


# def metric(obs1: float, obs2: float) -> float:
#     return abs(obs1 - obs2)


# K = 7
# THRESHOLD = 0.8
# INDENT_FACTOR = 0.25
# WINDOW_SIZE = 56
# SHIFT_FACTOR = 0.5
# SIGNIFICANCE_LEVEL = 0.05
# DATA_SIZE = 200
# DELTA = 5
# statistic = ThresholdOvercome(THRESHOLD)
# quality_metric = F1()
# scrubber = LinearScrubber(WINDOW_SIZE, SHIFT_FACTOR)
# ROOT_DIR = Path()
# RESULTS_DIR = ROOT_DIR / "experiments/results/f1/newsvm_norm/norm_8/normal-normal"
# DATASET_DIR = ROOT_DIR / "experiments/datasets/norm/norm_8"
# DEST_DIR = ROOT_DIR / "experiments/results/f1/newsvm_norm/"

# for alg_name in ["rf", "dt"]:
# if alg_name == "knn":
#     classifier = KNNAlgorithm(metric, statistic, INDENT_FACTOR, K)
# if alg_name == "rf":
#     classifier = RFClassifier()
# elif alg_name == "dt":
#     classifier = DecisionTreeClassifier()
# classifier = KNNAlgorithm(metric, statistic, INDENT_FACTOR, K)
# for i in range(1, 9):
#     DATASET_DIR = ROOT_DIR / f"experiments/datasets/norm/norm_{i}"
#     DEST_DIR = ROOT_DIR / f"experiments/results/f1/knn_norm/"
#     # cpd_alg = ClassificationAlgorithm(classifier, quality_metric, statistic, INDENT_FACTOR)
#     statistics_counting = StatisticsCalculation(classifier, scrubber)
#     statistics_counting.calculate_statistics(DATASET_DIR, DEST_DIR)


# RESULTS_DIR = ROOT_DIR / "experiments/results/f1/svm_norm/norm_8/normal-normal"
# cpd_data = LabeledCPData.read_generated_datasets(RESULTS_DIR)["normal-normal"].raw_data
# svm_cpd = CPDShell(cpd_data, cpd_algorithm=svm_algorithm, scrubber=scrubber)
# res_svm = svm_cpd.run_cpd()
# res_svm.visualize(True)
# print("SVM based algorithm")
# print(res_svm)

# classifier = SVMClassifier()
# cpd_alg = ClassificationAlgorithm(classifier, quality_metric, statistic, INDENT_FACTOR)

# statistics_counting = StatisticsCalculation(cpd_alg, scrubber)
# statistics_counting.calculate_statistics(DATASET_DIR, DEST_DIR)

# classifier = SVMClassifier()
# cpd_alg = ClassificationAlgorithm(classifier, quality_metric, statistic, INDENT_FACTOR)

# significance = ThresholdCalculation(
#     cpd_alg, scrubber, significance_level=SIGNIFICANCE_LEVEL, with_localization=True
# )

# DATASET_DIR = ROOT_DIR / "experiments/datasets/without_cp"
# # evaluated_threshold = significance.calculate_threshold_on_prepared_statistics(THRESHOLD, WINDOW_SIZE // 2, DATA_SIZE, DEST_DIR, INDENT_FACTOR, DELTA)
# evaluated_threshold = significance.calculate_threshold(THRESHOLD, DATASET_DIR)
# print(evaluated_threshold)


# stats_dirs = listdir(RESULTS_DIR)
# samples_count = len(stats_dirs)
# rates_sum = 0
# for stats_dir in stats_dirs:
#     rates_sum += Rates.true_positive_rate(DATA_SIZE // 2, RESULTS_DIR / stats_dir, statistic, DATA_SIZE, INDENT_FACTOR, WINDOW_SIZE, DELTA)

# print(rates_sum / samples_count)

# res_svm = svm_cpd.run_cpd()
# res_svm.visualize(True)
# print("SVM based algorithm")
# print(res_svm)
