from peal.operations.iteration import RandomStraightIteration
from peal.operations.operator import Operator


class ReproductionOperator(Operator):
    """Operator for the reproduction of individuals in a popoulation."""

    def __init__(
        self,
        in_individuals: int,
        out_individuals: int,
        probability: float,
    ):
        super().__init__(
            in_individuals,
            out_individuals,
            RandomStraightIteration(in_individuals, probability),
        )
