from peal.operations.iteration import NRandomBatchesIteration
from peal.operations.operator import Operator
from peal.population import Individual


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
        objects: tuple[Individual, ...],
    ) -> tuple[Individual, ...]:
        return (max(objects, key=lambda x: x.fitness).copy(), )


class Best(SelectionOperator):
    """A selection operator that returns the top ``out_size``
    individuals.
    """

    def __init__(self, in_size: int, out_size: int):
        if in_size < out_size:
            raise ValueError("in_size must at least be as big as out_size")
        super().__init__(
            in_size=in_size,
            out_size=out_size,
            iter_type=NRandomBatchesIteration(batch_size=in_size,),
        )

    def _process(
        self,
        objects: tuple[Individual, ...],
    ) -> tuple[Individual, ...]:
        return tuple(
            x.copy() for x in sorted(
                objects,
                key=lambda x: x.fitness,
                reverse=True,
            )
        )
