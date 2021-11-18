"""Module that implements common mutation operations."""

import numpy as np

from peal.population import Population
from peal.operations.config import operation
from peal.operations.iteration import every


@operation(category="mutation")
def flip_bits(
    population: Population,
    prob: float = 0.5,
) -> Population:
    iterator = every(population)
    result = Population()
    for ind in iterator:
        new_ind = ind.copy()
        for i in range(len(ind.genes)):
            if np.random.random_sample() <= prob:
                new_ind.genes[i] = not ind.genes[i]
        result.populate(new_ind)
    return result


@operation(category="mutation")
def uniform_int(
    population: Population,
    prob: float = 0.5,
    lowest: int = -1,
    highest: int = 1,
) -> Population:
    iterator = every(population)
    result = Population()
    for ind in iterator:
        new_ind = ind.copy()
        hits = np.where(np.random.random_sample(len(ind.genes)) <= prob)[0]
        new_ind.genes[hits] = np.random.randint(
            lowest,
            highest+1,
            size=len(hits)
        )
        result.populate(new_ind)
    return result
