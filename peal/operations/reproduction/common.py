"""Module that implements common reproduction operations."""

from typing import Union

import numpy as np

from peal.operations.reproduction.base import ReproductionOperator
from peal.population import Individual


class Crossover(ReproductionOperator):
    """Crossover reproduction operator.

    Args:
        npoints (int, optional): The number of points to use for the
            gene split in the crossover operation. Defaults to 2.
        probability (float): The probability of performing the crossover
            operation.
    """

    def __init__(self, npoints: int = 2, probability: float = 0.5):
        super().__init__(2, 2, probability)
        self._npoints = npoints

    def _process(
        self,
        individuals: Union[Individual, tuple[Individual, ...]],
    ) -> Union[Individual, tuple[Individual, ...]]:
        if not isinstance(individuals, tuple):
            raise TypeError("Crossover expects a tuple of individuals")

        points = np.insert(
            np.sort(
                np.random.randint(
                    1,
                    len(individuals[0].genes),
                    size=self._npoints
                )
            ),
            [0, -1],
            [0, len(individuals[0].genes)]
        )
        start = self._npoints % 2
        off1, off2 = individuals[0].copy(), individuals[1].copy()
        for i in range(start, self._npoints+(1-start), 2):
            off1.genes[points[i]:points[i+1]] = individuals[1].genes[
                points[i]:points[i+1]
            ]
            off2.genes[points[i]:points[i+1]] = individuals[0].genes[
                points[i]:points[i+1]
            ]
        return off1, off2
