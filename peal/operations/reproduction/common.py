"""Module that implements common reproduction operations."""

import numpy as np

from peal.operations.reproduction.base import ReproductionOperator
from peal.population import Population
from peal.operations.iteration import straight


class Crossover(ReproductionOperator):
    """Crossover reproduction operator.

    Args:
        prob (float, optional): The probability of a crossover
            happening. Defaults to 1.0.
        npoints (int, optional): The number of points to use for the
            gene split in the crossover operation. Defaults to 2.
    """

    def __init__(self, prob: float = 1.0, npoints: int = 2):
        self._prob = prob
        self._npoints = npoints

    def process(self, population: Population) -> Population:
        iterator = straight(population, batch_size=2)
        offspring = Population()
        for ind1, ind2 in iterator:
            off1 = ind1.copy()
            off2 = ind2.copy()
            if np.random.random_sample() <= self._prob:
                points = np.insert(
                    np.sort(
                        np.random.randint(
                            1,
                            len(ind1.genes),
                            size=self._npoints
                        )
                    ),
                    [0, -1],
                    [0, len(ind1.genes)]
                )
                start = self._npoints % 2
                for i in range(start, self._npoints+(1-start), 2):
                    off1.genes[points[i]:points[i+1]] = ind2.genes[
                        points[i]:points[i+1]
                    ]
                    off2.genes[points[i]:points[i+1]] = ind1.genes[
                        points[i]:points[i+1]
                    ]
            offspring.populate(off1, off2)
        return offspring
