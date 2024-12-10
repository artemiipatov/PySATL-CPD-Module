from os import listdir
from pathlib import Path

import numpy


def metric(obs1: float, obs2: float) -> float:
    return abs(obs1 - obs2)


# Directory config
ROOT_DIR = Path()
# DEST_DIR = f"experiments/results_k_3"
# SOURCE_DIR = "experiments/stage_1_knn"

# Sample config
SAMPLE_SIZE = 200
CP_LOCATION = 100

# Algorithm config
# K = 3
THRESHOLD = 1.2


for window_size in [72]:
    offset = int(window_size / 4)

    for K in [11]:
        for dir_n in ["beta_2", "exp_1", "exp_2"]:
            SOURCE_DIR = f"experiments/results_2nd_{window_size}_{K}_{dir_n}"
            RESULTS_DIR = ROOT_DIR / "experiments"
            distr_names = listdir(ROOT_DIR / f"{SOURCE_DIR}")
            distr_names.remove("info")

            false_positive = 0
            true_negative = 0
            false_negative = 0
            true_positive = 0

            for distr_name in distr_names:

                for num in range(100):
                    sample_name = ROOT_DIR / f"{SOURCE_DIR}/{distr_name}/sample_{num}"

                    with open(sample_name / "stats") as infile:
                        data = list(map(numpy.float64, infile.readlines()))

                    change_points = [i + offset for i, v in enumerate(data) if v > THRESHOLD]
                    central_change_points = [x for x in change_points if 100 - offset <= x <= 100 + offset]
                    side_change_points = [x for x in change_points if (x < 100 - offset or x > 100 + offset)]

                    if "-" in distr_name:
                        true_negative += len(data) - (len(side_change_points)) - (len(central_change_points) == 0)
                        false_positive += len(side_change_points)
                        false_negative += len(central_change_points) == 0
                        true_positive += len(central_change_points) != 0
                    else:
                        true_negative += len(data) - len(change_points)
                        false_positive += len(change_points)

            print(dir_n)
            print(
                f"window_size = {window_size}, K = {K}"
                + "\n"
                + f"FP Rate = {false_positive / (false_positive + true_negative)}"
            )
            print(f"FN Rate = {false_negative / (false_negative + true_positive)}")
