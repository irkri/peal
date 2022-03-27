from typing import Callable, Union

from peal.community import Community
from peal.population import Individual, Population


class Fitness:
    """Class that is responsible for calculating the fitness of
    individuals in an environment.

    Args:
        method (callable): The method to be called for evaluating a
            single individual. This method should expect a value of type
            :class:`~peal.individual.Individual` and return a float.
    """

    def __init__(self, method: Callable[[Individual], float]) -> None:
        self._method = method

    def evaluate(
        self,
        objects: Union[Individual, Population, Community],
    ) -> None:
        """Evaluates the fitness of individuals by changing their
        ``fitness`` attribute directly.

        Args:
            objects (Individual | Population | Community): A single
                individual, a population or a community to evaluate.
        """
        if isinstance(objects, Community):
            for pop in objects:
                for ind in pop:
                    ind.fitness = self._method(ind)
        elif isinstance(objects, Population):
            for ind in objects:
                ind.fitness = self._method(ind)
        elif isinstance(objects, Individual):
            objects.fitness = self._method(objects)
        else:
            raise TypeError(f"Cannot evaluate object of type {type(objects)}")

    def __call__(self, population: Union[Individual, Population]) -> None:
        self.evaluate(population)


def fitness(method: Callable[[Individual], float]) -> Fitness:
    """Decorator for a fitness method.

    Declaring your own fitness function is possible with the class
    :class:`~peal.fitness.Fitness` or using this decorator
    on your evaluation method.
    The method you want to decorate will need to have the same arguments
    and return types as described in the mentioned class.
    """
    return Fitness(method=method)
