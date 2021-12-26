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
            return individuals.copy()

        best = max(individuals, key=lambda x: x.fitness).copy()
        return best


class Best(SelectionOperator):
    """A selection operator that returns the top ``out_size``
    individuals.
    """

    def __init__(self, in_size: int, out_size: int):
        if in_size < out_size:
            raise ValueError("in_size must at least be as big as out_size")
        super().__init__(
            in_size,
            out_size,
            NRandomBatchesIteration(batch_size=in_size, total=1),
        )

    def _process(
        self,
        individuals: Union[Individual, tuple[Individual, ...]],
    ) -> Union[Individual, tuple[Individual, ...]]:
        if isinstance(individuals, Individual):
            return individuals.copy()

        sorted_individuals = tuple(
            x.copy() for x in sorted(
                individuals,
                key=lambda x: x.fitness,
                reverse=True,
            )
        )
        return sorted_individuals[:self._out_size]
