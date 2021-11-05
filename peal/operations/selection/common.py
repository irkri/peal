"""Module that implements common selection operations."""

from peal.environment.population import Population
from peal.operations.config import operation
from peal.operations.iteration import random_batches


@operation(category="selection")
def tournament(
    population: Population,
    size: int = 2,
) -> Population:
    iterator = random_batches(population, n=population.size, size=size)
    result = Population()
    for tourn in iterator:
        best = max(tourn, key=lambda x: x.fitness)
        result.populate(best.copy())
    return result
