"""Module that implements common reproduction operations."""

import numba
import numpy as np

from peal.population import Population
from peal.operations.config import operation
from peal.operations.iteration import straight


@numba.njit(cache=True)
def _crossover_2_point(
    g1: np.ndarray,
    g2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    point1 = np.random.randint(0, len(g1))
    point2 = np.random.randint(0, len(g1))
    if point1 > point2:
        point1, point2 = point2, point1
    if point1 == point2:
        return g1, g2
    temp = g1[point1:point2]
    g1[point1:point2] = g2[point1:point2]
    g2[point1:point2] = temp
    return g1, g2


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
            if npoints == 2:
                off1.genes, off2.genes = _crossover_2_point(
                    off1.genes, off2.genes
                )
            else:
                points = np.insert(
                    np.sort(
                        np.random.randint(1, len(ind1.genes), size=npoints)
                    ),
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
