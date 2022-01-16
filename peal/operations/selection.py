import numpy as np

from peal.operations.iteration import (
    NRandomBatchesIteration,
    StraightIteration,
)
from peal.operations.operator import Operator, PopulationOperator
from peal.individual import Individual
from peal.population import Population


class SelectionOperator(Operator):
    """Operator for the selection of individuals from a popoulation."""


class Tournament(SelectionOperator):
    """A selection operator that simulates a tournament selection of
    variable size.

    Args:
        size (int, optional): The number of individuals participating in
            the tournament.
    """

    def __init__(self, size: int = 2):
        super().__init__(
            in_size=size,
            out_size=1,
            iter_type=NRandomBatchesIteration(batch_size=size),
        )

    def _process(
        self,
        individuals: tuple[Individual, ...],
    ) -> tuple[Individual, ...]:
        return (max(individuals, key=lambda x: x.fitness).copy(), )


class Best(SelectionOperator):
    """A selection operator that returns the top ``out_size``
    individuals of a population.
    """

    def __init__(self, in_size: int, out_size: int):
        if in_size < out_size:
            raise ValueError("in_size must at least be as big as out_size")
        super().__init__(
            in_size=in_size,
            out_size=out_size,
            iter_type=StraightIteration(batch_size=in_size),
        )

    def _process(
        self,
        individuals: tuple[Individual, ...],
    ) -> tuple[Individual, ...]:
        return tuple(
            x.copy() for x in sorted(
                individuals,
                key=lambda x: x.fitness,
                reverse=True,
            )[:self._out_size]
        )


class BestMean(PopulationOperator):
    """Operator that selects populations out of a tuple of populations
    based on their highest mean fitness.

    Args:
        in_size (int): Number of populations to input.
        out_size (int): Number of populations to select. It has to hold
            that ``in_size >= out_size``.
    """

    def __init__(self, in_size: int, out_size: int):
        super().__init__(in_size=in_size, out_size=out_size)

    def _process(
        self,
        populations: tuple[Population, ...],
    ) -> tuple[Population, ...]:
        return tuple(pop.deepcopy() for pop in sorted(
            populations,
            key=lambda pop: np.mean(pop.fitness),
            reverse=True,
        )[:self._out_size])
