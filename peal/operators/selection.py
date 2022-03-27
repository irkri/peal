"""Module that provides operators that select individuals from a
population or populations from a community.
"""

import numpy as np

from peal.community import Community
from peal.operators.iteration import NRandomBatchesIteration, StraightIteration
from peal.operators.operator import Operator
from peal.population import Population


class Tournament(Operator):
    """A selection operator that simulates a tournament selection of
    variable size for multiple individuals in a population.

    Args:
        size (int, optional): The number of individuals participating in
            a single tournament.
    """

    def __init__(self, size: int = 2) -> None:
        super().__init__(iter_type=NRandomBatchesIteration(batch_size=size))

    def _process_population(
        self,
        container: Population,
    ) -> Population:
        return Population(max(container, key=lambda x: x.fitness).copy())


class Best(Operator):
    """A selection operator that returns the top ``out_size``
    individuals of a population.
    """

    def __init__(self, in_size: int, out_size: int) -> None:
        if in_size < out_size:
            raise ValueError("in_size must at least be as big as out_size")
        super().__init__(iter_type=StraightIteration(batch_size=in_size))
        self._out_size = out_size

    def _process_population(
        self,
        container: Population,
    ) -> Population:
        return Population([ind.copy() for ind in sorted(
            container,
            key=lambda x: x.fitness,
            reverse=True,
        )[:self._out_size]])


class BestMean(Operator):
    """Operator that selects populations out of a community based on
    their highest mean fitness.

    Args:
        in_size (int): Number of populations to select from.
        out_size (int): Number of populations to select.
    """

    def __init__(self, in_size: int, out_size: int) -> None:
        super().__init__(iter_type=StraightIteration(batch_size=in_size))
        self._out_size = out_size

    def _process_community(
        self,
        container: Community,
    ) -> Community:
        return Community([pop.deepcopy() for pop in sorted(
            container,
            key=lambda pop: np.mean(pop.fitness),
            reverse=True,
        )[:self._out_size]])
