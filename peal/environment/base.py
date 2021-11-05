"""Module that defines the default environment for a evolutionary
algorithm.
"""

from typing import Union

from peal.environment.population import Population
from peal.evaluation.fitness import Fitness
from peal.individual.base import Individual
from peal.operations.config import _Toperationundetermined, check_marked


class Environment:
    """The default environment for an evolutionary algorithm."""

    def __init__(self, fitness: Fitness):
        self._population: Population = Population()
        self._operations: dict[str, _Toperationundetermined] = dict()
        self._generation: int = 0
        self._fitness = fitness

    def push(self, *individuals: Union[Individual, Population]):
        """Pushes the given individuals or populations to the
        environment.

        Args:
            individuals (Individual | Population): One or more
                individuals or populations to push.
        """
        for ind in individuals:
            if isinstance(ind, Individual):
                self._population.populate(ind)
            elif isinstance(ind, Population):
                self._population.populate(*ind)
            else:
                raise TypeError(f"Cannot push objects of type {type(ind)} "
                                "to environment")

    def use(self, *operations: _Toperationundetermined):
        """Mark operations to be used in the environment. This may
        override already given operations.

        Args:
            operations: One or more peal operations, i.e. decorated
                functions. Available decorators are defined in
                :module:`peal.operations`.
        """
        for operation in operations:
            marked_as = check_marked(operation)
            self._operations[marked_as] = operation

    def evolve(self, ngen: int):
        """Start an evolutionary process.

        Args:
            ngen (int): Number of generations to evolve.
        """
        if "selection" not in self._operations:
            raise RuntimeError("No selection operation specified.")
        for _ in range(ngen):
            offspring = Population()
            self._fitness.evaluate(self._population)
            selected = self._operations["selection"](
                self._population
            )
            offspring.populate(self._operations["reproduction"](
                selected
            ))
            self._population = self._operations["mutation"](
                offspring
            )
            self._generation += 1
