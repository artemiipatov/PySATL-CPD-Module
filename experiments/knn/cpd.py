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
THRESHOLD = 4


for window_size in [48]:
    offset = int(window_size / 4)

    for K in range(9, 10, 2):
        SOURCE_DIR = f"experiments/results_{window_size}_{K}"
        RESULTS_DIR = ROOT_DIR / "experiments"
        distr_names = listdir(ROOT_DIR / f"{SOURCE_DIR}")
        distr_names.remove("info")

        with open(RESULTS_DIR / f"cpd_{window_size}_{K}", "w+") as outfile:
            for distr_name in distr_names:
                outfile.write(distr_name + "\n")

                for num in range(3):
                    sample_name = ROOT_DIR / f"{SOURCE_DIR}/{distr_name}/sample_{num}"

                    with open(sample_name / "stats") as infile:
                        data = list(map(numpy.float64, infile.readlines()))

                    change_points = [i + offset for i, v in enumerate(data) if v > THRESHOLD]

                    # Save result to file
                    outfile.write(" ".join(map(str, change_points)) + "\n")
