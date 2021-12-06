"""Module that implements common selection operations."""

from typing import Union
from peal.operations.iteration import NRandomBatchesIteration

from peal.operations.selection.base import SelectionOperator
from peal.population import Individual


class Tournament(SelectionOperator):
    """A selection operator that simulates a tournament selection of
    variable size.

    Args:
        size (int, optional): The number of individuals participating in
            the tournament.
    """

    def __init__(self, size: int = 2):
        super().__init__(
            size,
            1,
            NRandomBatchesIteration(batch_size=size),
        )

    def _process(
        self,
        individuals: Union[Individual, tuple[Individual, ...]],
    ) -> Union[Individual, tuple[Individual, ...]]:
        if isinstance(individuals, Individual):
            return individuals

        best = max(individuals, key=lambda x: x.fitness).copy()
        return best
