from os import listdir
from pathlib import Path
from statistics import fmean

import numpy
from matplotlib import pyplot as plt


def metric(obs1: float, obs2: float) -> float:
    return abs(obs1 - obs2)


# Directory config
ROOT_DIR = Path()
# DEST_DIR = f"experiments/results_k_3"
SOURCE_DIR = "experiments/stage_1_knn"

# Sample config
SAMPLE_SIZE = 200
CP_LOCATION = 100

# Algorithm config
# K = 3
THRESHOLD = 4.7

for K in range(5, 6, 2):
    DEST_DIR = f"experiments/results_32_{K}"
    distr_names = listdir(ROOT_DIR / f"{SOURCE_DIR}")
    distr_names.remove("experiment_description")

    RESULTS_DIR = ROOT_DIR / f"{DEST_DIR}"

    for distr_name in distr_names:
        distr_dir = ROOT_DIR / DEST_DIR / f"{distr_name}"

        for num in range(3):
            sample_dir = distr_dir / f"sample_{num}"

            # Save result to file
            with open(sample_dir / "stats") as infile:
                data = list(map(numpy.float64, infile.readlines()))

            averaging: list[numpy.float64] = []
            without_center = data[:84:] + data[100:]
            average = fmean(without_center)
            result_array = [average] * 84 + data[84:100] + [average] * (176 - 100)

            # Save plot
            plt.plot(result_array)
            plt.title(f"{distr_name}/sample_{num}")
            plt.xlabel("Time")
            plt.ylabel("Statistics")
            plt.vlines(x=92, ymin=min(data), ymax=max(data), colors="orange", ls="--")
            # remove(sample_dir / f"avg.png")
            plt.savefig(sample_dir / "difference.png")
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
