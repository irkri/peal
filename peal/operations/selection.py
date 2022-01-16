import numpy as np

from peal.community import Community
from peal.operations.iteration import (
    NRandomBatchesIteration,
    StraightIteration,
)
from peal.operations.operator import Operator
from peal.population import Population


class Tournament(Operator[Population]):
    """A selection operator that simulates a tournament selection of
    variable size for multiple individuals in a population.

    Args:
        size (int, optional): The number of individuals participating in
            the tournament.
    """

    def __init__(self, size: int = 2):
        super().__init__(
            iter_type=NRandomBatchesIteration[Population](batch_size=size)
        )

    def _process(
        self,
        container: Population,
    ) -> Population:
        return Population(max(container, key=lambda x: x.fitness).copy())


class Best(Operator[Population]):
    """A selection operator that returns the top ``out_size``
    individuals of a population.
    """

    def __init__(self, in_size: int, out_size: int):
        if in_size < out_size:
            raise ValueError("in_size must at least be as big as out_size")
        super().__init__(
            iter_type=StraightIteration[Population](batch_size=in_size),
        )
        self._out_size = out_size

    def _process(
        self,
        container: Population,
    ) -> Population:
        return Population([ind.copy() for ind in sorted(
            container,
            key=lambda x: x.fitness,
            reverse=True,
        )[:self._out_size]])


class P_BestMean(Operator[Community]):
    """Operator that selects populations out of a community of
    populations based on their highest mean fitness.

    Args:
        in_size (int): Number of populations to select from.
        out_size (int): Number of populations to select.
    """

    def __init__(self, in_size: int, out_size: int):
        super().__init__(
            iter_type=StraightIteration[Community](batch_size=in_size)
        )
        self._out_size = out_size

    def _process(
        self,
        container: Community,
    ) -> Community:
        return Community([pop.deepcopy() for pop in sorted(
            container,
            key=lambda pop: np.mean(pop.fitness),
            reverse=True,
        )[:self._out_size]])
