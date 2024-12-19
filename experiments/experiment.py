from pathlib import Path
from os import listdir

from CPDShell.Core.scrubber.linear_scrubber import LinearScrubber
from CPDShell.Core.scrubber_scenario import ScrubberScenario
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
from experiments.significance import ThresholdCalculation
from experiments.rates import Rates
from CPDShell.labeled_data import LabeledCPData


def metric(obs1: float, obs2: float) -> float:
    return abs(obs1 - obs2)


K = 7
THRESHOLD = 0.8
INDENT_FACTOR = 0.25
WINDOW_SIZE = 56
SHIFT_FACTOR = 0.5
SIGNIFICANCE_LEVEL = 0.05
DATA_SIZE = 200
DELTA = 5
statistic = ThresholdOvercome(THRESHOLD)
quality_metric = F1()
scrubber = LinearScrubber(WINDOW_SIZE, SHIFT_FACTOR)
ROOT_DIR = Path()
RESULTS_DIR = ROOT_DIR / "experiments/results/f1/newsvm_norm/norm_8/normal-normal"
DATASET_DIR = ROOT_DIR / "experiments/datasets/norm/norm_8"
DEST_DIR = ROOT_DIR / "experiments/results/f1/newsvm_norm/"

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

classifier = SVMClassifier()
cpd_alg = ClassificationAlgorithm(classifier, quality_metric, statistic, INDENT_FACTOR)

significance = ThresholdCalculation(
    cpd_alg, scrubber, significance_level=SIGNIFICANCE_LEVEL, with_localization=True
)

DATASET_DIR = ROOT_DIR / "experiments/datasets/without_cp"
# evaluated_threshold = significance.calculate_threshold_on_prepared_statistics(THRESHOLD, WINDOW_SIZE // 2, DATA_SIZE, DEST_DIR, INDENT_FACTOR, DELTA)
evaluated_threshold = significance.calculate_threshold(THRESHOLD, DATASET_DIR)
print(evaluated_threshold)


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