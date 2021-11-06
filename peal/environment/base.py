"""Module that defines the default environment for a evolutionary
algorithm.
"""

from typing import Union

from peal.environment.population import Population
from peal.evaluation.callback import Callback
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

    def compile(self):
        if "selection" not in self._operations:
            raise RuntimeError("No selection operation specified.")
        if "reproduction" not in self._operations:
            raise RuntimeError("No reproduction operation specified.")
        if "mutation" not in self._operations:
            raise RuntimeError("No mutation operation specified.")

    def evolve(self, ngen: int, callbacks: list[Callback]):
        """Start an evolutionary process.

        Args:
            ngen (int): Number of generations to evolve.
        """
        self.compile()
        for callback in callbacks:
            if not isinstance(callback, Callback):
                raise TypeError("All callbacks have to be of type Callback")
            callback.on_start(self._population)
        for _ in range(ngen):
            self._fitness.evaluate(self._population)
            for callback in callbacks:
                callback.on_generation_start(self._population)
            offspring = Population()
            selected = self._operations["selection"](
                self._population
            )
            for callback in callbacks:
                callback.on_selection(selected)
            offspring.populate(self._operations["reproduction"](
                selected
            ))
            for callback in callbacks:
                callback.on_reproduction(offspring)
            self._population = self._operations["mutation"](
                offspring
            )
            for callback in callbacks:
                callback.on_mutation(self._population)
            self._generation += 1
            for callback in callbacks:
                callback.on_generation_end(self._population)
        self._fitness.evaluate(self._population)
        for callback in callbacks:
            callback.on_end(self._population)
