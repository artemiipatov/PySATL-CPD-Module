from os import listdir
from pathlib import Path

import numpy
from matplotlib import pyplot as plt


def metric(obs1: float, obs2: float) -> float:
    return abs(obs1 - obs2)


# Directory config
ROOT_DIR = Path()
# DEST_DIR = f"experiments/results_k_3"
# SOURCE_DIR = "experiments/results"

# Sample config
SAMPLE_SIZE = 200
CP_LOCATION = 100

# Algorithm config
# K = 3
THRESHOLD = 4.7

# for K in [7]:
#     for i in [1, 2, 3, 4]:
#         for window_size in [40, 48]:
# DEST_DIR = f"experiments/results_2nd_{window_size}_{K}_exp_{i}"
for dir_n in [
    "exp_1",
    "exp_2",
    "exp_3",
    "exp_4",
    "norm_1",
    "norm_2",
    "norm_3",
    "norm_4",
    "norm_5",
    "norm_6",
    "norm_7",
    "norm_8",
    "norm_9",
    "beta_1",
    "beta_2",
    "beta_3",
    "beta_4",
    "uniform_1",
    "uniform_2",
    "uniform_3",
]:
    DEST_DIR = f"experiments/stage_2_knn_{dir_n}"
    distr_names = listdir(ROOT_DIR / f"{DEST_DIR}")
    if len(distr_names) > 1:
        distr_names.remove("experiment_description")

    RESULTS_DIR = ROOT_DIR / f"{DEST_DIR}"

    for distr_name in distr_names:
        distr_dir = ROOT_DIR / DEST_DIR / f"{distr_name}"

        for num in range(20):
            # sample_dir = distr_dir / f"sample_{num}"
            sample_dir = distr_dir / f"sample_{num}/{distr_name}"

            # Save result to file
            with open(sample_dir / "sample.csv") as infile:
                data = list(map(numpy.float64, infile.readlines()))

            # Save plot
            plt.plot(data)
            plt.title(f"{distr_name}/sample_{num}")
            plt.xlabel("Time")
            plt.ylabel("Data")
            plt.vlines(x=100, ymin=min(data), ymax=max(data), colors="orange", ls="--")
            # remove(sample_dir / f"avg.png")
            # plt.show()
            plt.savefig(sample_dir / "plot.png")
            plt.clf()


# sample_name = ROOT_DIR / f"{SOURCE_DIR}/normal-normal/sample_{0}"

# # Run algorithm
# cpd_data = LabeledCPData.read_generated_datasets(sample_name)['normal-normal'].raw_data
# shell = CPDShell(cpd_data)
# shell.scenario.change_point_number = 100
# shell.CPDalgorithm = KNNAlgorithm(metric, k=5, threshold=THRESHOLD)
# change_points = shell.run_cpd()
# print(change_points)
# change_points.visualize()
