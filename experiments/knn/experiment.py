from pathlib import Path
from experiments.knn.significance import KNNSignificance
from CPDShell.Core.algorithms.ClassificationBasedCPD.classifiers.knn.knn_classifier import KNNAlgorithm
from CPDShell.Core.algorithms.ClassificationBasedCPD.test_statistics.threshold_overcome import ThresholdOvercome
from CPDShell.Core.algorithms.classification_algorithm import ClassificationAlgorithm


def metric(obs1: float, obs2: float) -> float:
    return abs(obs1 - obs2)


ROOT_DIR = Path()
DATASET_DIR = ROOT_DIR / f"experiments/without_cp/normal/"

K = 5
THRESHOLD = 3.0
OFFSET_COEFF = 0.25
WINDOW_SIZE = 32
MOVEMENT_K = 0.5
SIGNIFICANCE_LEVEL = 0.05

knn_classifier = KNNAlgorithm(metric, K)
statistic = ThresholdOvercome(THRESHOLD)
knn_algorithm = ClassificationAlgorithm(knn_classifier, statistic, OFFSET_COEFF)
significance = KNNSignificance(knn_algorithm,
                               WINDOW_SIZE,
                               MOVEMENT_K,
                               significance_level = SIGNIFICANCE_LEVEL,
                               with_localization=True)

evaluated_threshold = significance.calculate_threshold(THRESHOLD, DATASET_DIR)
print(evaluated_threshold)

