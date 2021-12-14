"""Module that defines the default environment for a evolutionary
algorithm.
"""

import re
from typing import Optional

import numpy as np

from peal.core.integration import (
    IntegrationTechnique,
    OffspringFirstIntegration,
)

from peal.evaluation import Callback, Fitness

from peal.operations import (
    MutationOperator,
    ReproductionOperator, MultiMix,
    SelectionOperator, Best,
)

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
        selection (SelectionOperator): The selection operation that will
            be used.
        reproduction (ReproductionOperator): The reproduction operation
            that will be used.
        mutation (MutationOperator, optional): A mutation operation
            that can be specified but doesn't have to. Defaults to None.
        integration (IntegrationTechnique): The integration technique
            that integrates offspring created by the reproduction
            operator in the population.
            Defaults to :class:`OffspringFirstIntegration`.
    """

    def __init__(
        self,
        breeder: Breeder,
        fitness: Fitness,
        selection: SelectionOperator,
        reproduction: ReproductionOperator,
        mutation: Optional[MutationOperator] = None,
        integration: Optional[IntegrationTechnique] = None,
    ):
        self._selection = selection
        self._reproduction = reproduction
        self._mutation = mutation
        if integration is not None:
            self._integration = integration
        else:
            self._integration = OffspringFirstIntegration()
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
                Defaults to None.

        Returns:
            Population: A population after created in the last
                generation.
        """
        if callbacks is None:
            callbacks = []
        for callback in callbacks:
            if not isinstance(callback, Callback):
                raise TypeError("Callback has unexpected type "
                                f"{type(callback)}")

        population = Population()
        population.populate(self._breeder.breed(population_size))
        self._fitness.evaluate(population)
        for callback in callbacks:
            callback.on_start(population)

        for _ in range(ngen):

            for callback in callbacks:
                callback.on_generation_start(population)

            offspring = Population()
            selected = self._selection(population)

            offspring.populate(self._reproduction(selected))
            population = self._integration.merge(
                offspring,
                selected,  # type: ignore
            )

            if self._mutation is not None:
                population = self._mutation(population)  # type: ignore

            self._fitness.evaluate(population)
            self._generation += 1
            for callback in callbacks:
                callback.on_generation_end(population)

        for callback in callbacks:
            callback.on_end(population)

        return population


class StrategyProcess:
    r"""A process that follows a certain evolutionary strategy given by
    the argument ``signature`` for a number of generations.

    Args:
        breeder (Breeder): The breeder to use for population
            initialization.
        fitness (Fitness): The fitness to use for evaluation.
        mutation (MutationOperator, optional): A mutation operation
            that can be specified but doesn't have to. Defaults to None.
        signature (str, optional): The notation Schwefel (1977) used to
            characterize multimembered evolutionary strategies. This is
            a string matching the (pythonic) regular expression
            ``"\((\d+)(/\d+)?(\+|,)(\d+)\)"``.
    """

    signature_regex: str = r"\((\d+)(/\d+)?(\+|,)(\d+)\)"

    def __init__(
        self,
        breeder: Breeder,
        fitness: Fitness,
        mutation: Optional[MutationOperator] = None,
        signature: str = "",
    ):
        self._breeder = breeder
        self._fitness = fitness
        self._mutation = mutation
        self._initialize_parameters(signature)

    def _initialize_parameters(self, signature: str):
        match = re.search(self.signature_regex, signature)
        if match is None:
            raise ValueError("Given signature does not match the "
                             "required pattern")
        signature_matches = match.groups()
        self._parents_included: bool = signature_matches[2] == "+"
        # mu: number of parents
        self._ind_mu: int = int(signature_matches[0])
        self._pop_mu: int = 1
        # lambda: number of offspring
        self._ind_lambda: int = int(signature_matches[3])
        self._pop_lambda: int = 1
        # rho: mixin
        self._ind_rho: int = int(signature_matches[1][1:])
        self._pop_rho: int = 1
        # gamma: cycle number
        self._ind_gamma: int = 1
        self._pop_gamma: int = 1
        # operators
        self._selection = Best(
            in_size=self._ind_lambda,
            out_size=self._ind_mu,
        )
        self._reproduction = MultiMix(in_size=self._ind_rho)

    def start(
        self,
        ngen: int,
        callbacks: Optional[list[Callback]] = None,
    ) -> Population:
        """Starts the strategy process."""
        if callbacks is None:
            callbacks = []
        for callback in callbacks:
            if not isinstance(callback, Callback):
                raise TypeError("Callback has unexpected type "
                                f"{type(callback)}")

        population = Population()
        population.populate(self._breeder.breed(self._ind_mu))
        self._fitness.evaluate(population)
        for callback in callbacks:
            callback.on_start(population)

        for _ in range(ngen):

            for callback in callbacks:
                callback.on_generation_start(population)

            selected_parent_indices = [
                np.random.randint(
                    0,
                    population.size,
                    size=self._ind_rho,
                ) for _ in range(self._ind_lambda)
            ]
            offspring = Population()
            for indices in selected_parent_indices:
                if self._ind_rho == 1:
                    offspring.populate(population[indices[0]])
                else:
                    offspring.populate(
                        self._reproduction(
                            tuple(population[i] for i in indices)
                        )
                    )

            if self._mutation is not None:
                offspring = self._mutation(offspring)  # type: ignore
            self._fitness.evaluate(offspring)
            if self._parents_included:
                offspring.populate(population)
            population = Population(self._selection(tuple(offspring)))

            for callback in callbacks:
                callback.on_generation_end(population)

        return population
