from pathlib import Path

from CPDShell.Core.scrubber.linear_scrubber import LinearScrubber
from CPDShell.Core.scrubber_scenario import ScrubberScenario
from CPDShell.Core.algorithms.classification_algorithm import ClassificationAlgorithm
from CPDShell.Core.algorithms.ClassificationBasedCPD.classifiers.svm.svm_classifier import SVMClassifier
from CPDShell.Core.algorithms.ClassificationBasedCPD.quality_metrics.classification.mcc import MCC
from CPDShell.Core.algorithms.ClassificationBasedCPD.test_statistics.threshold_overcome import ThresholdOvercome
from CPDShell.shell import CPDShell
from experiments.significance import ThresholdCalculation
from CPDShell.labeled_data import LabeledCPData


def metric(obs1: float, obs2: float) -> float:
    return abs(obs1 - obs2)


# ROOT_DIR = Path()
# SOURCE_DIR = f"experiments/stage_2_knn_norm_6"
# DISTR_NAME = "normal-normal"
# sample_dir = ROOT_DIR / SOURCE_DIR / f"{DISTR_NAME}/sample_9"
# cpd_data = LabeledCPData.read_generated_datasets(sample_dir)[DISTR_NAME].raw_data


ROOT_DIR = Path()
DATASET_DIR = ROOT_DIR / "experiments/without_cp/normal/"


K = 5
THRESHOLD = 0.3
INDENT_COEFF = 0.25
WINDOW_SIZE = 56
SHIFT_FACTOR = 0.5
SIGNIFICANCE_LEVEL = 0.05

svm_classifier = SVMClassifier()
statistic = ThresholdOvercome(THRESHOLD)
quality_metric = MCC()
scrubber = LinearScrubber(WINDOW_SIZE, SHIFT_FACTOR)
svm_algorithm = ClassificationAlgorithm(svm_classifier, quality_metric, statistic, INDENT_COEFF)
# svm_cpd = CPDShell(cpd_data, cpd_algorithm=knn_algorithm, scrubber=scrubber)

# res_svm = svm_cpd.run_cpd()
# res_svm.visualize(True)
# print("SVM based algorithm")
# print(res_svm)

significance = ThresholdCalculation(
    svm_algorithm, scrubber, significance_level=SIGNIFICANCE_LEVEL, with_localization=True
)

evaluated_threshold = significance.calculate_threshold(THRESHOLD, DATASET_DIR)
print(evaluated_threshold)
