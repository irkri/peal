"""Module that defines the base breeder class."""

from typing import Callable

from peal.population.population import Population
from peal.population.individual import Individual


class Breeder:
    """Class that manages the generation of new
    :class:`~peal.individual.base.Individual` instances.

    Args:
        method (callable): A method that returns an
            :class:`~peal.individual.base.Individual`.
    """

    def __init__(self, method: Callable[[], Individual]):
        self._method = method

    def breed(self, size: int) -> Population:
        """Returns a population of given size using the given method.

        Args:
            size (int): Number of individuals to breed.
        """
        population = Population()
        for _ in range(size):
            population.populate(self._method())
        return population

    def __call__(self) -> Population:
        return self.breed()


def breeder(method: Callable[[], Individual]) -> Breeder:
    """Decorator for a breeding method.

    Declaring your own breeder is possible with the class
    :class:`~peal.environment.breeder.Breeder` or using this decorator
    on your breeding method.
    The method you want to decorate will need to have the same arguments
    and return types as described in the mentioned class.
    """
    return Breeder(method=method)
