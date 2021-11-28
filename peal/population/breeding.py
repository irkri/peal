"""Module that defines the base breeder class."""

from typing import Callable

import numpy as np

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

    def breed(self, size: int = 1) -> Population:
        """Returns a population of given size using the given method.

        Args:
            size (int, optional): Number of individuals to breed.
                Defaults to 1.
        """
        population = Population()
        for _ in range(size):
            population.populate(self._method())
        return population

    def __call__(self, size: int = 1) -> Population:
        return self.breed(size)


def breeder(method: Callable[[], Individual]) -> Breeder:
    """Decorator for a breeding method.

    Declaring your own breeder is possible with the class
    :class:`~peal.environment.breeder.Breeder` or using this decorator
    on your breeding method.
    The method you want to decorate will need to have the same arguments
    and return types as described in the mentioned class.
    """
    return Breeder(method=method)


class IntegerBreeder(Breeder):
    """Breeder that creates individuals with random integer genes placed
    in a one dimensional numpy array.

    Args:
        size (int): Number of genes the resulting individual will have.
        lower (int, optional): The lowest number a gene of this type can
            have (included). Defaults to 0.
        upper (int, optional): The highest number a gene of this type
            can have (included). Defaults to 1.
    """

    def __init__(
        self,
        size: int,
        lower: int = 0,
        upper: int = 1,
    ):
        super().__init__(
            lambda: Individual(np.random.randint(lower, upper+1, size))
        )
