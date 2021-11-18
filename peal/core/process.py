"""Module that defines the default environment for a evolutionary
algorithm.
"""

from typing import Optional

from peal.population import Breeder, Population
from peal.evaluation.callback import Callback
from peal.evaluation.fitness import Fitness
from peal.operations.config import _Toperation, check_operation


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
        selection: _Toperation,
        reproduction: _Toperation,
        mutation: Optional[_Toperation] = None,
    ):
        if check_operation(selection) != "selection":
            raise TypeError(f"Not a selection operation: {selection!r}")
        self._selection = selection
        if check_operation(reproduction) != "reproduction":
            raise TypeError(f"Not a reproduction operation: {reproduction!r}")
        self._reproduction = reproduction
        if mutation is not None and check_operation(mutation) != "mutation":
            raise TypeError(f"Not a mutation operation: {mutation!r}")
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
        if callbacks is not None:
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
