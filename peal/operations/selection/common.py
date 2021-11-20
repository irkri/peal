"""Module that implements common selection operations."""

from peal.operations.selection.base import SelectionOperator
from peal.population import Population
from peal.operations.iteration import random_batches


class Tournament(SelectionOperator):
    """A selection operator that simulates a tournament selection of
    variable size.

    Args:
        size (int, optional): The number of individuals participating in
            the tournament.
    """

    def __init__(self, size: int = 2):
        self._size = size

    def process(self, population: Population) -> Population:
        iterator = random_batches(
            population,
            total=population.size,
            size=self._size
        )
        result = Population()
        for tourn in iterator:
            best = max(tourn, key=lambda x: x.fitness)
            result.populate(best.copy())
        return result
