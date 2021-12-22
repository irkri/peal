from peal.operations.iteration import SingleIteration
from peal.operations.operator import Operator


class MutationOperator(Operator):
    """Operator for the mutation of individuals in a popoulation."""

    def __init__(
        self,
        in_individuals: int,
        out_individuals: int,
    ):
        super().__init__(
            in_individuals,
            out_individuals,
            SingleIteration(),
        )
