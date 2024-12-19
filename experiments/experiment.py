from pathlib import Path
from os import listdir
import yaml

from CPDShell.Core.scrubber.linear_scrubber import LinearScrubber
from CPDShell.Core.scrubber_scenario import ScrubberScenario
from CPDShell.Core.scrubber.abstract_scrubber import Scrubber
from CPDShell.Core.algorithms.classification_algorithm import ClassificationAlgorithm
from CPDShell.Core.algorithms.ClassificationBasedCPD.classifiers.rf.rf_classifier import RFClassifier
from CPDShell.Core.algorithms.ClassificationBasedCPD.classifiers.svm.svm_classifier import SVMClassifier
from CPDShell.Core.algorithms.ClassificationBasedCPD.classifiers.decision_tree.decision_tree_classifier import DecisionTreeClassifier
from CPDShell.Core.algorithms.ClassificationBasedCPD.abstracts.iclassifier import Classifier
from CPDShell.Core.algorithms.knn_algorithm import KNNAlgorithm
from CPDShell.Core.algorithms.ClassificationBasedCPD.quality_metrics.classification.mcc import MCC
from CPDShell.Core.algorithms.ClassificationBasedCPD.quality_metrics.classification.f1 import F1
from CPDShell.Core.algorithms.ClassificationBasedCPD.test_statistics.threshold_overcome import ThresholdOvercome
from CPDShell.shell import CPDShell
from experiments.statistics_calculation import StatisticsCalculation
from experiments.threshold_calculation import ThresholdCalculation
from experiments.rates import Rates
from CPDShell.labeled_data import LabeledCPData
import experiments.generator as gen


class Experiments():
    def __init__(self, cpd_algorithm: ClassificationAlgorithm | KNNAlgorithm, scrubber: Scrubber, config_path: Path, dataset_path: Path, results_path: Path) -> None:
        self.__config_path = config_path
        self.__dataset_path = dataset_path
        self.__results_path = results_path

        self.__cpd_algorithm = cpd_algorithm
        self.__scrubber = scrubber
        self.__distributions: list[gen.DistributionComposition] = []
    
    def generate(self, sample_count: int):
        with open(self.__config_path) as stream:
            loaded_config = yaml.safe_load(stream)

        for distr_comp_conf in loaded_config:
            distr_comp: gen.DistributionComposition = [gen.Distribution(d_conf["type"], d_conf["parameters"], d_conf["length"]) for d_conf in distr_comp_conf]
            self.__distributions.append(distr_comp)

        gen.DistributionGenerator.generate(self.__distributions, sample_count, self.__dataset_path)

    def run(self, significance_level: float, with_localization: bool = True, delta: float = 0.005):
        # generate without cp first
        threshold_calculation = ThresholdCalculation(self.__cpd_algorithm, self.__scrubber, significance_level, with_localization, delta)
        threshold = threshold_calculation.calculate_threshold(1.0, self.__dataset_path)
        statistics_calculation = StatisticsCalculation(self.__cpd_algorithm, self.__scrubber)
        statistics_calculation.calculate_statistics(self.__dataset_path, self.__results_path)
        # print(f"FNR: {Rates.false_negative_rate()}")

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
