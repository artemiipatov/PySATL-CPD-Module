import csv
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field

import yaml

from CPDShell.generator.generator import ScipyDatasetGenerator
from CPDShell.generator.saver import DatasetSaver

# SAMPLE_SIZE = 200
# CP_LOCATION = 100

# NUM_OF_SAMPLES = 250

# DIR_NAME = f"normal_{NUM_OF_SAMPLES}"
# DIR_PATH = f"/experiments/datasets/without_cp/{DIR_NAME}/"
# CONFIG_NAME = "config.yml"

# WORKING_DIR = Path()


class VerboseSafeDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


class DistributionType(Enum):
    NORMAL = 1
    EXPONENTIAL = 2
    UNIFORM = 3
    WEIBULL = 4
    BETA = 5


@dataclass
class Distribution:
    type: DistributionType
    parameters: dict[str, float]
    length: int

DistributionComposition = list[Distribution]


class DistributionGenerator():
    # def __init__(self, ) -> None:
    #     self.__distributions = distributions

    @staticmethod
    def generate(distributions: list[DistributionComposition], sample_count: int, dest_path: Path):
        Path(dest_path).mkdir(parents=True, exist_ok=True)
        distributions_info = DistributionGenerator.__generate_configs(distributions, sample_count, dest_path)
        DistributionGenerator.__generate_experiment_description(distributions_info, dest_path)
        DistributionGenerator.__generate_dataset(distributions_info, dest_path)

    @staticmethod
    def __generate_configs(distributions: list[DistributionComposition], sample_count: int, dest_path: Path) -> list[tuple[str, int]]:
        generated_distributions_info = []

        for distributionComp in distributions:
            name = "-".join(map(lambda d: d.type.name, distributionComp))
            generated_distributions_info.append((name, sample_count))

            config = [
                {
                    "name": name,
                    "distributions": [{"type": d_conf.type, "length": d_conf.length, "parameters": d_conf.parameters} for d_conf in distributionComp]
                }
            ]

            Path(dest_path / name).mkdir(parents=True, exist_ok=True)
            with open(dest_path / f"{name}/config.yaml", "w") as outfile:
                yaml.dump(config, outfile, default_flow_style=False, sort_keys=False, Dumper=VerboseSafeDumper)

    @staticmethod
    def __generate_experiment_description(distributions_info: list[tuple[str, int]], dest_path: Path):
        with open(dest_path / "experiment_description", "w", newline="") as f:
            write = csv.writer(f)
            write.writerow(["name", "samples_num"])
            samples_description = [[d_info[0], str(d_info[1])] for d_info in distributions_info]
            write.writerows(samples_description)

    @staticmethod
    def __generate_dataset(distributions_info: list[tuple[str, int]], dest_path: Path):
        for d_info in distributions_info:
            name = d_info[0]
            sample_count = d_info[1]

            Path(dest_path / name).mkdir(parents=True, exist_ok=True)

            for sample_num in range(sample_count):
                print(f"Name: {name}. Sample num: {sample_num}")
                Path(dest_path / f"{name}/sample_{sample_num}/").mkdir(
                    parents=True, exist_ok=True
                )
                saver = DatasetSaver(dest_path / f"{name}/sample_{sample_num}/", True)
                ScipyDatasetGenerator().generate_datasets(
                    Path(dest_path / f"{name}/config.yaml"), saver
                )

                Path(
                    dest_path / f"{name}/sample_{sample_num}/{name}/sample.png"
                ).unlink(missing_ok=True)
