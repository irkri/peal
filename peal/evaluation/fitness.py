"""Module that provides the class
:class:`~peal.evaluation.fitness.Fitness` which is responsible for
calculating the fitness of single individuals in an environment.
"""

from typing import Callable
from peal.environment.population import Population
from peal.individual.base import Individual


class Fitness:
    """Class that is responsible for calculating the fitness of
    individuals in an environment.

    Args:
        method (callable): The method to be called for evaluating a
            single individual. This method should expect a value of
            type :class:`~peal.individual.base.Individual` and
            return a ``float``.
    """

    def __init__(self, method: Callable[[Individual], float]):
        self._method = method

    def evaluate(self, population: Population):
        """Evaluates the fitness of all individuals in the given
        population. This method changes the values in ``population``
        directly.


        Args:
            population (Population): Population to evaluate.

        Returns:
            list[float]: A list of fitness values for each individual.
        """
        for ind in population:
            ind.fitness = self._method(ind)
