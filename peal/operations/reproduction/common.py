"""Module that implements common reproduction operations."""

import numpy as np

from peal.population import Population
from peal.operations.config import operation
from peal.operations.iteration import straight


@operation(category="reproduction")
def crossover(
    population: Population,
    prob: float = 1.0,
    npoints: int = 2,
) -> Population:
    iterator = straight(population, batch_size=2)
    offspring = Population()
    for ind1, ind2 in iterator:
        off1 = ind1.copy()
        off2 = ind2.copy()
        if np.random.random_sample() <= prob:
            points = np.insert(
                np.sort(np.random.randint(1, len(ind1.genes), size=npoints)),
                [0, -1],
                [0, len(ind1.genes)]
            )
            start = npoints % 2
            for i in range(start, npoints+(1-start), 2):
                off1.genes[points[i]:points[i+1]] = ind2.genes[
                    points[i]:points[i+1]
                ]
                off2.genes[points[i]:points[i+1]] = ind1.genes[
                    points[i]:points[i+1]
                ]
        offspring.populate(off1, off2)
    return offspring
