from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import re
from typing import ClassVar, Optional

import numpy as np

from peal.breeding import Breeder
from peal.core.callback import Callback
from peal.fitness import Fitness
from peal.operations.mutation import MutationOperator, NormalDist
from peal.operations.reproduction import ReproductionOperator, MultiMix
from peal.operations.selection import SelectionOperator, Best
from peal.operations.integration import IntegrationOperator, OffspringFirst
from peal.population import Population


class _AbstractProcess(ABC):

    @abstractmethod
    def start(self, callbacks: Optional[list[Callback]] = None) -> None:
        """Starts the evolutionary process.

        Args:
            callbacks (list[Callback], optional): A number of
                :class:`~peal.core.callback.Callback` objects that can
                be used to track the status of the evolution. If a
                callback is suitable for this process may be noted in
                the classes docstring.
        """


@dataclass
class _AbstractProcessData:

    breeder: Breeder
    fitness: Fitness


class Process(_AbstractProcess, _AbstractProcessData):
    """Abstract class for an evolutionary process.

    An evolutionary process is a class in PEAL that describes a certain
    kind of algorithm or approach to an optimization problem that is
    related to the theory of evolutionary algorithms.
    Each of these processes can be customized based on the information
    given in the corresponding class.

    Args:
        breeder (Breeder): The breeder to use for population
            initialization.
        fitness (Fitness): The fitness to use for evaluation, i.e. the
            target function.
    """


@dataclass
class SynchronousProcess(Process):
    """This synchronous process (also called generational process)
    mimics the most popular genetic algorithm that uses selection,
    mutation and reproduction operations to manipulate an existing
    population for each new generation.

    Args:
        breeder (Breeder): The breeder to use for population
            initialization.
        fitness (Fitness): The fitness to use for evaluation, i.e. the
            target function.
        init_size (int): The initial size of the population at start of
            the evolution.
        generations (int): The number of generations this process will
            be repeated on the initial population.
        selection (SelectionOperator): The selection operation that will
            be used.
        reproduction (ReproductionOperator): The reproduction operation
            that will be used.
        mutation (MutationOperator, optional): A mutation operation
            that can be specified but doesn't have to. Defaults to None.
        integration (IntegrationOperator, optional): The operation that
            integrates offspring created by the reproduction operator
            into the parent population of the new generation. Defaults
            to :class:`~peal.operations.integration.OffspringFirst`.
    """

    init_size: int
    generations: int
    selection: SelectionOperator
    reproduction: ReproductionOperator
    mutation: Optional[MutationOperator] = None
    integration: IntegrationOperator = field(
        default_factory=OffspringFirst
    )

    def start(self, callbacks: Optional[list[Callback]] = None) -> None:
        callbacks = [] if callbacks is None else callbacks

        population = Population()
        population.populate(self.breeder.breed(self.init_size))
        self.fitness.evaluate(population)
        for callback in callbacks:
            callback.on_start(population)

        for _ in range(self.generations):

            for callback in callbacks:
                callback.on_generation_start(population)

            offspring = Population()
            selected = self.selection.process(population)

            offspring.populate(self.reproduction.process(selected))
            population = self.integration.process(
                (offspring, selected)  # type: ignore
            )[0]

            if self.mutation is not None:
                population = self.mutation.process(
                    population,
                )  # type: ignore

            self.fitness.evaluate(population)
            for callback in callbacks:
                callback.on_generation_end(population)

        for callback in callbacks:
            callback.on_end(population)


@dataclass
class StrategyProcess(Process):
    """A process that follows a certain evolutionary strategy given by
    the argument ``signature`` for a number of generations.

    Args:
        breeder (Breeder): The breeder to use for population
            initialization.
        fitness (Fitness): The fitness to use for evaluation.
        generations (int): The number of generations this process will
            be repeated on the initial population.
        signature (str): The notation Schwefel (1977) used to
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
        alpha (float, optional): The alpha value to use for mutation
            step size change. In every generation, a single individual
            has its own mutation step size. This is a float that gets
            multiplied with a normally distributed number to simulate
            mutation (this multiplication result then gets added to each
            gene of the individual).
    """

    generations: int
    signature: str
    alpha: float = 1.3
    signature_regex: ClassVar[str] = (
        r"(?:(\d+)(/\d+)?(\+|,)(\d+))?\((\d+)(/\d+)?(\+|,)(\d+)\)(\^\d+)?"
    )

    def __post_init__(self):
        self._initialize_parameters(self.signature)

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
        self._ind_gamma: int = 1 if matches[8] is None else int(matches[8][1:])
        # operators
        self._selection = Best(
            in_size=(
                self._ind_lambda + self._ind_mu if self._ind_parents_included
                else self._ind_lambda
            ),
            out_size=self._ind_mu,
        )
        self._reproduction = MultiMix(in_size=self._ind_rho)

    def start(self, callbacks: Optional[list[Callback]] = None):
        callbacks = [] if callbacks is None else callbacks

        # initialize self._pop_mu populations of size self._ind_mu
        populations: list[Population] = []
        for i in range(self._pop_mu):
            populations.append(Population())
            populations[-1].populate(self.breeder.breed(self._ind_mu))
            for j, indiv in enumerate(populations[-1]):
                indiv.hidden_genes = np.array([1], dtype=np.float32)
            self.fitness.evaluate(populations[-1])
            for callback in callbacks:
                callback.on_start(populations[-1])

        for _ in range(self.generations):
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
                parts = [self._ind_mu // self._pop_rho
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

                    # create offspring for each population
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
                        offspring[-1].hidden_genes = population[
                            np.random.choice(indices)
                        ].hidden_genes

                    # apply a mutation to each individual
                    for j, individual in enumerate(offspring):
                        individual.hidden_genes *= (
                            np.random.choice([self.alpha, 1/self.alpha])
                        )
                        offspring[j] = NormalDist(
                            1.0, 0, individual.hidden_genes[0]**2
                        ).process((individual, ))[0]

                    # select self._ind_mu individuals
                    self.fitness.evaluate(offspring)
                    if self._ind_parents_included:
                        offspring.populate(population)
                    offspring_populations[i] = self._selection(
                        offspring
                    )  # type: ignore

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
