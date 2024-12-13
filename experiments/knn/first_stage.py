from os import listdir
from pathlib import Path
from statistics import fmean

from matplotlib import pyplot as plt

from CPDShell.Core.algorithms.classification_algorithm import ClassificationAlgorithm
from CPDShell.Core.algorithms.KNNCPD.knn_classifier import KNNAlgorithm
from CPDShell.Core.algorithms.ClassificationBasedCPD.test_statistics.threshold_overcome import ThresholdOvercome
from CPDShell.labeled_data import LabeledCPData
from CPDShell.shell import CPDShell


def metric(obs1: float, obs2: float) -> float:
    return abs(obs1 - obs2)


# # Directory config
ROOT_DIR = Path()
# DEST_DIR = f"experiments/results_k_3"


# Sample config
# SAMPLE_SIZE = 200
# CP_LOCATION = 100

# # Algorithm config
# # K = 3
THRESHOLD = 3.0
# for dir_n in ["exp_1", "exp_2", "beta_2"]:
SOURCE_DIR = "experiments/"

# for K in [5]:
for N in [56]:
    offset_coeff = 0.25
    offset = N * offset_coeff
    DEST_DIR = f"experiments/results_without_cp_{N}_{K}"
    distr_names = listdir(ROOT_DIR / f"{SOURCE_DIR}")

    if len(distr_names) > 1:
        distr_names.remove("experiment_description")
    # distr_names.remove('beta')
    # distr_names.remove('beta-uniform')
    # distr_names.remove('beta-normal')
    # distr_names.remove('beta-weibull')
    # distr_names.remove('weibull')
    # distr_names.remove('weibull-exponential')
    # distr_names.remove('weibull-weibull')
    # distr_names.remove('weibull-normal')
    # distr_names.remove('weibull-uniform')
    # distr_names.remove('normal-beta')
    # distr_names.remove('normal-exponential')
    # distr_names.remove('normal-normal')
    # distr_names.remove('normal-uniform')
    # distr_names.remove('normal-weibull')
    # distr_names.remove('normal')
    # distr_names.remove('exponential')
    # distr_names.remove('exponential-beta')
    # distr_names.remove('exponential-uniform')
    # distr_names.remove('exponential-weibull')
    # distr_names.remove('exponential-normal')
    # distr_names.remove('uniform-exponential')
    # distr_names.remove('uniform-uniform')
    # distr_names.remove('uniform-normal')
    # distr_names.remove('uniform-beta')

    RESULTS_DIR = ROOT_DIR / f"{DEST_DIR}"

    Path(ROOT_DIR / f"{DEST_DIR}").mkdir(parents=True, exist_ok=True)

    time_list: list[float] = []

    for distr_name in distr_names:
        distr_dir = ROOT_DIR / DEST_DIR / f"{distr_name}"
        Path(distr_dir).mkdir(parents=True, exist_ok=True)

        for num in range(50):
            sample_name = ROOT_DIR / f"{SOURCE_DIR}/{distr_name}/sample_{num}"
            sample_dir = distr_dir / f"sample_{num}"

            # Create folder for current result
            sample_dir = distr_dir / f"sample_{num}"
            Path(sample_dir).mkdir(parents=True, exist_ok=True)

            # Run algorithm
            cpd_data = LabeledCPData.read_generated_datasets(sample_name)[distr_name].raw_data

            knn_classifier = KNNAlgorithm(metric, K)
            statistic = ThresholdOvercome(THRESHOLD)
            knn_algorithm = ClassificationAlgorithm(knn_classifier, statistic, offset_coeff)
            shell = CPDShell(cpd_data, knn_algorithm)

            shell.scenario.change_point_number = 100
            shell.scrubber.window_length = N
            shell.scrubber.movement_k = 0.5
            change_points = shell.run_cpd()
            print(change_points)
            time_list.append(change_points.time_sec)

            stats = shell.CPDalgorithm.statistics_list

            # Save result to file
            with open(sample_dir / "stats", "w+") as outfile:
                for stat in stats:
                    outfile.write(str(stat) + "\n")

            # Save plot
            plt.plot(stats)
            plt.title(f"{distr_name}/sample_{num}")
            plt.xlabel("Time")
            plt.ylabel("Statistics")
            # plt.vlines(x=(100 - offset), ymin=min(stats), ymax=max(stats), colors="orange", ls="--")
            plt.savefig(sample_dir / "plot.png")
            plt.clf()

    with open(ROOT_DIR / f"{DEST_DIR}/info", "w+") as outfile:
        outfile.write(str(fmean(time_list)) + "\n")


# sample_name = ROOT_DIR / f"{SOURCE_DIR}/normal-normal/sample_{0}"

# # Run algorithm
# cpd_data = LabeledCPData.read_generated_datasets(sample_name)['normal-normal'].raw_data
# shell = CPDShell(cpd_data)
# shell.scenario.change_point_number = 100
# shell.scrubber.window_length = 40
# shell.scrubber.movement_k = 0.5
# shell.CPDalgorithm = KNNAlgorithm(metric, k=5, threshold=THRESHOLD)
# change_points = shell.run_cpd()
# print(change_points)
# change_points.visualize()
