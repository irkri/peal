"""Module that provides the class
:class:`~peal.evaluation.fitness.Fitness` which is responsible for
calculating the fitness of single individuals in an environment.
"""

from typing import Callable, Union

from peal.population import Individual, Population


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

    def evaluate(self, population: Union[Individual, Population]):
        """Evaluates the fitness of all individuals in the given
        population. This method changes attributes of individuals in
        ``population`` directly.

        Args:
            population (Population | Individual): The Population or a
                single individual to evaluate.

        Returns:
            list[float]: A list of fitness values for each individual.
        """
        if isinstance(population, Population):
            for ind in population:
                if not ind.fitted:
                    ind.fitness = self._method(ind)
        elif isinstance(population, Individual):
            population.fitness = self._method(population)

    def __call__(self, population: Union[Individual, Population]):
        self.evaluate(population)


def fitness(method: Callable[[Individual], float]) -> Fitness:
    """Decorator for a fitness method.

    Declaring your own fitness function is possible with the class
    :class:`~peal.evaluation.fitness.Fitness` or using this decorator
    on your evaluation method.
    The method you want to decorate will need to have the same arguments
    and return types as described in the mentioned class.
    """
    return Fitness(method=method)
