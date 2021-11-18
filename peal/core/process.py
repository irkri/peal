"""Module that defines the default environment for a evolutionary
algorithm.
"""

from typing import Optional

from peal.evaluation.callback import Callback
from peal.evaluation.fitness import Fitness
from peal.operations.mutation.base import MutationOperator
from peal.operations.reproduction.base import ReproductionOperator
from peal.operations.selection.base import SelectionOperator
from peal.population import Breeder, Population


class SynchronousProcess:
    """This synchronous process (also called generational process)
    mimics the most popular genetic algorithm that uses selection,
    mutation and reproduction operations to manipulate an existing
    population for each new generation.

    Args:
        breeder (Breeder): The breeder to use for population
            initialization.
        fitness (Fitness): The fitness to use for evaluation.
        selection (_Toperation): The selection operation that will be
            used.
        reproduction (_Toperation): The reproduction operation that will
            be used.
        mutation (_Toperation, optional): A mutation operation
            that can be specified but doesn't have to. Defaults to None.
    """

    def __init__(
        self,
        breeder: Breeder,
        fitness: Fitness,
        selection: SelectionOperator,
        reproduction: ReproductionOperator,
        mutation: Optional[MutationOperator] = None,
    ):
        self._selection = selection
        self._reproduction = reproduction
        self._mutation = mutation
        self._breeder = breeder
        self._fitness = fitness
        self._generation: int = 0

    def start(
        self,
        population_size: int,
        ngen: int,
        callbacks: Optional[list[Callback]] = None,
    ) -> Population:
        """Start the evolutionary process.

        Args:
            population_size (int): Number of individuals to start with.
            ngen (int): Number of generations to evolve in one call of
                ``self.start()``.
            callbacks (list[Callback], optional): Optional list of
                callbacks that are used in the process.

        Returns:
            Population: A population after the last generation finished.
        """
        if callbacks is None:
            callbacks = []
        for callback in callbacks:
            if not isinstance(callback, Callback):
                raise TypeError("Callback has unexpected type "
                                f"{type(callback)}")
        population = Population()
        population.populate(self._breeder.breed(population_size))
        for callback in callbacks:
            callback.on_start(population)

        for _ in range(ngen):
            self._fitness.evaluate(population)

            for callback in callbacks:
                callback.on_generation_start(population)

            offspring = Population()
            selected = self._selection(population)
            for callback in callbacks:
                callback.on_selection(selected)

            offspring.populate(self._reproduction(selected))
            for callback in callbacks:
                callback.on_reproduction(offspring)

            if self._mutation is not None:
                population = self._mutation(offspring)
            for callback in callbacks:
                callback.on_mutation(population)

            self._generation += 1
            for callback in callbacks:
                callback.on_generation_end(population)

        self._fitness.evaluate(population)
        for callback in callbacks:
            callback.on_end(population)

        return population
