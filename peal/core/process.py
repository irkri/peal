"""An evolutionary process is a class in PEAL that describes a certain
kind of algorithm or approach to an optimization problem that is related
to the theory of evolutionary algorithms.
Each of these processes can be customized based on the information given
in the corresponding class.
"""

import re
from typing import Optional

import numpy as np

from peal.evaluation import Callback, Fitness

from peal.operations.mutation import MutationOperator
from peal.operations.reproduction import ReproductionOperator, MultiMix
from peal.operations.selection import SelectionOperator, Best

from peal.population import (
    Breeder,
    Population,
    IntegrationTechnique,
    OffspringFirstIntegration,
)


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
            operator into the population.
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
            ngen (int): Number of generations to evolve.
            callbacks (list[Callback], optional): Optional list of
                callbacks that are used in the process.
                Defaults to None.

        Returns:
            Population: A population created in the last generation.
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
    """A process that follows a certain evolutionary strategy given by
    the argument ``signature`` for a number of generations.

    Args:
        breeder (Breeder): The breeder to use for population
            initialization.
        fitness (Fitness): The fitness to use for evaluation.
        mutation (MutationOperator, optional): A mutation operation
            that can be specified but doesn't have to. Defaults to None.
        signature (str, optional): The notation Schwefel (1977) used to
            characterize multimembered evolutionary strategies. This is
            a string matching the expression ``a/b{,|+}c(d/e{,|+}f)^g``
            where ``{,|+}`` denotes the character ``,`` or ``+``. Other
            symbols are used to describe the following.

            * ``a``: Number of parent populations.
            * ``b``: Number of populations to use for creating one new
              population.
            * ``c``: Number of offspring populations.
            * ``d``: Number of parent individuals.
            * ``e``: Number of individuals to use from one parent
              population to reproduce offspring.
            * ``f``: Size of one offspring population.
            * ``g``: Number of iterations each population evolves before
              mixin them.

            The values ``a``, ``b`` and ``c`` are optional, ``a`` and
            ``c`` always have to be supplied if one of them is
            specified. Also ``b``, ``e`` and ``g`` are optional.
            Specifying ``d+f`` refers to the strategy of using also
            the parent individuals of one population for selecting the
            offspring while ``d,f`` only selects individuals from the
            mutated and reproduced original population. The same applies
            for ``a+c`` and ``a,c``, only on population level.
    """

    signature_regex: str = (
        r"(?:(\d+)(/\d+)?(\+|,)(\d+))?\((\d+)(/\d+)?(\+|,)(\d+)\)(\^\d+)?"
    )

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
        # evaluates the given signature of the evolutionary strategy
        match = re.search(self.signature_regex, signature)
        if match is None:
            raise ValueError("Given signature does not match the "
                             "required pattern")
        matches = match.groups()
        self._ind_parents_included: bool = matches[6] == "+"
        self._pop_parents_included: bool = matches[2] == "+"
        # mu: number of parents
        self._ind_mu: int = int(matches[4])
        self._pop_mu: int = 1 if matches[0] is None else int(matches[0])
        # lambda: number of offspring
        self._ind_lambda: int = int(matches[7])
        self._pop_lambda: int = 1 if matches[3] is None else int(matches[3])
        # rho: mixin proportion number
        self._ind_rho: int = 1 if matches[5] is None else int(matches[5][1:])
        self._pop_rho: int = 1 if matches[1] is None else int(matches[1][1:])
        # gamma: cycle number for single population evolution
        self._ind_gamma: int = 1 if matches[0] is None else int(matches[8][1:])
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
    ) -> list[Population]:
        """Start the evolutionary process.

        Args:
            ngen (int): Number of generations to evolve, that is,
                applying the given evolutionary strategy on one or more
                populations.
            callbacks (list[Callback], optional): Optional list of
                callbacks that are used in the process. Here, the
                methods of the callbacks are called for each generation
                on individuals level. Meaning that they are called
                ``ngen*c*g`` times, where ``c`` and ``g`` refer to
                variables used in this classes docstring.
                Defaults to None.

        Returns:
            list[Population]: A list of populations created in the final
                generation.
        """
        if callbacks is None:
            callbacks = []
        for callback in callbacks:
            if not isinstance(callback, Callback):
                raise TypeError("Callback has unexpected type "
                                f"{type(callback)}")

        # initialize self._pop_mu populations of size self._ind_mu
        populations: list[Population] = []
        for _ in range(self._pop_mu):
            populations.append(Population())
            populations[-1].populate(self._breeder.breed(self._ind_mu))
            self._fitness.evaluate(populations[-1])
            for callback in callbacks:
                callback.on_start(populations[-1])

        for _ in range(ngen):
            # create self._pop_lambda new populations
            # to do so, take self._pop_rho populations to create a
            # single new one by throwing a similar proportion of
            # individuals from these populations together
            offspring_populations: list[Population] = []
            population_parent_indices = [
                np.random.randint(
                    0,
                    self._pop_mu,
                    size=self._pop_rho,
                ) for _ in range(self._pop_lambda)
            ]
            for indices in population_parent_indices:
                new_population = Population()
                parts = [self._ind_mu//self._pop_rho
                         for _ in range(self._pop_rho)]
                for i in range(self._ind_mu % self._pop_rho):
                    parts[i % self._pop_rho] += 1
                for i in indices:
                    for j in parts:
                        new_population.populate(populations[i][0:j])
                offspring_populations.append(new_population)

            # now evolve each population according to the supplied
            # evolutionary strategy on individuals level
            for _ in range(self._ind_gamma):
                for i, population in enumerate(offspring_populations):
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
                    if self._ind_parents_included:
                        offspring.populate(population)
                    offspring_populations[i] = Population(
                        self._selection(tuple(offspring))
                    )

                    for callback in callbacks:
                        callback.on_generation_end(offspring_populations[i])

            # next, select self._pop_mu populations according to
            # their mean fitness
            if self._pop_parents_included:
                for population in populations:
                    offspring_populations.append(population)
            populations = sorted(
                offspring_populations,
                key=lambda population: np.mean(population.fitness),
                reverse=True,
            )[:self._pop_mu]

        return populations
