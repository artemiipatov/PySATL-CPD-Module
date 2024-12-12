from pathlib import Path

from CPDShell.Core.algorithms.bayesian_algorithm import BayesianAlgorithm
from CPDShell.Core.algorithms.BayesianCPD.detectors.drop_detector import DropDetector
from CPDShell.Core.algorithms.BayesianCPD.detectors.simple_detector import SimpleDetector
from CPDShell.Core.algorithms.BayesianCPD.hazards.constant_hazard import ConstantHazard
from CPDShell.Core.algorithms.BayesianCPD.likelihoods.gaussian_unknown_mean_and_variance import (
    GaussianUnknownMeanAndVariance,
)
from CPDShell.Core.algorithms.BayesianCPD.localizers.simple_localizer import SimpleLocalizer
from CPDShell.Core.algorithms.ClassificationBasedCPD.test_statistics.threshold_overcome import ThresholdOvercome
from CPDShell.Core.algorithms.ClassificationBasedCPD.quality_metrics.classification.f1 import F1
from CPDShell.Core.algorithms.ClassificationBasedCPD.classifiers.svm.svm_classifier import SVMClassifier
from CPDShell.Core.algorithms.ClassificationBasedCPD.classifiers.rf.rf_classifier import RFClassifier
from CPDShell.Core.algorithms.classification_algorithm import ClassificationAlgorithm
from CPDShell.Core.algorithms.knn_algorithm import KNNAlgorithm
from CPDShell.generator.generator import ScipyDatasetGenerator
from CPDShell.generator.saver import DatasetSaver
from CPDShell.shell import CPDShell
from CPDShell.labeled_data import LabeledCPData

# path_string = "tests/test_CPDShell/test_configs/test_config_exp.yml"
# distributions_name = "exp"

# saver = DatasetSaver(Path(), True)
# generated = ScipyDatasetGenerator().generate_datasets(Path(path_string), saver)
# data, expected_change_points = generated[distributions_name]

# print("Expected change points:", expected_change_points)

# # Graph algorithm demo
# graph_cpd = CPDShell(data)
# graph_cpd.scrubber.window_length = 150
# graph_cpd.scrubber.movement_k = 2.0 / 3.0

# res_graph = graph_cpd.run_cpd()
# res_graph.visualize(True)
# print("Graph algorithm")
# print(res_graph)


# # k-NN based algorithm demo
# def metric(obs1: float, obs2: float) -> float:
#     return abs(obs1 - obs2)


# K = 5
# KNN_THRESHOLD = 3.5
# OFFSET_COEFF = 0.25

# statistic = ThresholdOvercome(KNN_THRESHOLD)
# knn_algorithm = KNNAlgorithm(metric, statistic, OFFSET_COEFF, K)
# knn_cpd = CPDShell(data, knn_algorithm)

# knn_cpd.scrubber.window_length = 32
# knn_cpd.scrubber.movement_k = 0.5
# knn_cpd.scenario.change_point_number = 100

# res_knn = knn_cpd.run_cpd()
# res_knn.visualize(True)
# print("k-NN based algorithm")
# print(res_knn)

# path_string = "tests/test_CPDShell/test_configs/test_config_exp.yml"
# distributions_name = "exp"

# saver = DatasetSaver(Path(), True)
# generated = ScipyDatasetGenerator().generate_datasets(Path(path_string), saver)
# data, expected_change_points = generated[distributions_name]

# print("Expected change points:", expected_change_points)


ROOT_DIR = Path()
SOURCE_DIR = f"experiments/stage_2_knn"
DISTR_NAME = "beta-weibull"
sample_dir = ROOT_DIR / SOURCE_DIR / f"{DISTR_NAME}/sample_0"
cpd_data = LabeledCPData.read_generated_datasets(sample_dir)[DISTR_NAME].raw_data

SVM_THRESHOLD = 0.858
SVM_OFFSET_COEFF = 0.25

svm_classifier = SVMClassifier()
statistic = ThresholdOvercome(SVM_THRESHOLD)
quality_metric = F1()
svm_algorithm = ClassificationAlgorithm(svm_classifier, quality_metric, statistic, SVM_OFFSET_COEFF)
svm_cpd = CPDShell(cpd_data, svm_algorithm)

svm_cpd.scrubber.window_length = 48
svm_cpd.scrubber.movement_k = 0.5
svm_cpd.scenario.change_point_number = 50

res_svm = svm_cpd.run_cpd()
res_svm.visualize(True)
print("SVM based algorithm")
print(res_svm)


# ROOT_DIR = Path()
# SOURCE_DIR = f"experiments/stage_2_knn"
# DISTR_NAME = "normal-normal"
# sample_dir = ROOT_DIR / SOURCE_DIR / f"{DISTR_NAME}/sample_0"
# cpd_data = LabeledCPData.read_generated_datasets(sample_dir)[DISTR_NAME].raw_data

# RF_THRESHOLD = 0.76
# RF_OFFSET_COEFF = 0.25

# rf_algorithm = RFClassifier()
# statistic = ThresholdOvercome(RF_THRESHOLD)
# quality_metric = F1()
# rf_algorithm = ClassificationAlgorithm(rf_algorithm, quality_metric, statistic, RF_OFFSET_COEFF)
# rf_cpd = CPDShell(cpd_data, rf_algorithm)

# rf_cpd.scrubber.window_length = 48
# rf_cpd.scrubber.movement_k = 0.5
# rf_cpd.scenario.change_point_number = 200

# res_rf = rf_cpd.run_cpd()
# res_rf.visualize(True)
# print("RF based algorithm")
# print(res_rf)



# # Bayesian algorithm demo
# BAYESIAN_THRESHOLD = 0.1
# NUM_OF_SAMPLES = 1000
# SAMPLE_SIZE = 500
# BERNOULLI_PROB = 1.0 - 0.5 ** (1.0 / SAMPLE_SIZE)
# HAZARD_RATE = 1 / BERNOULLI_PROB
# LEARNING_SAMPLE_SIZE = 50
# BAYESIAN_DROP_THRESHOLD = 0.7

# constant_hazard = ConstantHazard(HAZARD_RATE)
# gaussian_likelihood = GaussianUnknownMeanAndVariance()

# simple_detector = SimpleDetector(BAYESIAN_THRESHOLD)
# drop_detector = DropDetector(BAYESIAN_DROP_THRESHOLD)

# simple_localizer = SimpleLocalizer()

# bayesian_algorithm = BayesianAlgorithm(
#     learning_steps=LEARNING_SAMPLE_SIZE,
#     likelihood=gaussian_likelihood,
#     hazard=constant_hazard,
#     detector=simple_detector,
#     localizer=simple_localizer,
# )

# bayesian_cpd = CPDShell(data, bayesian_algorithm)
# bayesian_cpd.scrubber.window_length = 500
# bayesian_cpd.scrubber.movement_k = 2.0 / 3.0

# res_bayes = bayesian_cpd.run_cpd()
# res_bayes.visualize(True)
# print("Bayesian algorithm")
# print(res_bayes)
