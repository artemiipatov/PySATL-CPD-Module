"""
Module for implementation of neareset neighbours graph.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

import typing as tp
from collections import deque
from collections.abc import Iterable

import numpy as np

from .abstracts.observation import Observation
from .knn_heap import NNHeap


class KNNGraph:
    """
    The class implementing nearest neighbours graph.
    """

    def __init__(
        self,
        window: Iterable[float],
        metric: tp.Callable[[float, float], float],
        k=7,
        delta=1e-12,
    ) -> None:
        """
        Initializes a new instance of KNN graph.

        :param window: an overall sample the graph is based on.
        :param metric: function for calculating the distance between two points in time series.
        :param k: number of neighbours in the knn graph relative to each point.
        Default is 7, which is generally the most optimal value (based on the experiments results).
        :param delta: delta for comparing float values of the given observations.
        """
        self.__window: list[Observation] = [Observation(t, v) for t, v in enumerate(window)]
        self.__metric: tp.Callable[[Observation, Observation], float] = lambda obs1, obs2: metric(
            obs1.value, obs2.value
        )
        self.__k = k
        self.__delta = delta

        self.__graph: deque[NNHeap] = deque(maxlen=len(self.__window))

    def build(self) -> None:
        """
        Builds KNN graph according to the given parameters.
        """
        for i in range(len(self.__window)):
            heap = NNHeap(self.__k, self.__metric, self.__window[-i - 1], self.__delta)
            heap.build(self.__window)
            self.__graph.appendleft(heap)

    def get_neighbours(self, obs_index: int) -> list[int]:
        return self.__graph[obs_index].get_neighbours_indices()

    def check_for_neighbourhood(self, first_index: int, second_index: int) -> bool:
        """
        Checks if the second observation is among the k nearest neighbours of the first observation.

        :param first_index: index of main observation.
        :param second_index: index of possible neighbour.
        :return: true if the second point is the neighbour of the first one, false otherwise.
        """
        neighbour = self.__window[second_index]
        return self.__graph[first_index].find_in_heap(neighbour)
