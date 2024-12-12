import pytest

from CPDShell.Core.algorithms.graph_algorithm import GraphAlgorithm
from CPDShell.Core.cpd_core import CPDCore
from CPDShell.Core.scrubber.linear_scrubber import LinearScrubber
from CPDShell.Core.scrubber_scenario import ScrubberScenario


def custom_comparison(node1, node2):
    arg = 5
    return abs(node1 - node2) <= arg


class TestCPDCore:
    @pytest.mark.parametrize(
        "scenario_param,data,alg_class,alg_param,scrubber_data_size,expected",
        (
            (
                (1, True),
                (50, 55, 60, 48, 52, 70, 75, 80, 90, 85, 95, 100, 50),
                GraphAlgorithm,
                (custom_comparison, 1.5),
                100,
                [5],
            ),
            (
                (1, False),
                (50, 55, 60, 48, 52, 70, 75, 80, 90, 85, 95, 100, 50),
                GraphAlgorithm,
                (custom_comparison, 1.5),
                100,
                [0],
            ),
        ),
    )
    def test_run(self, scenario_param, data, alg_class, alg_param, scrubber_data_size, expected):
        scenario = ScrubberScenario(*scenario_param)
        scrubber = LinearScrubber()
        algorithm = alg_class(*alg_param)

        core = CPDCore(scenario, data, scrubber, algorithm, scrubber_data_size)
        assert core.run() == expected
