from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import re
from typing import ClassVar, Optional

from peal.breeding import Breeder
from peal.core.callback import Callback
from peal.fitness import Fitness
from peal.operations.iteration import NRandomBatchesIteration
from peal.operations.mutation import MutationOperator, NormalDist
from peal.operations.reproduction import (
    EquiMix, ReproductionOperator, MultiMix
)
from peal.operations.selection import BestMean, SelectionOperator, Best
from peal.operations.integration import IntegrationOperator, FirstThingsFirst
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
            to :class:`~peal.operations.integration.FirstThingsFirst`.
    """

    init_size: int
    generations: int
    selection: SelectionOperator
    reproduction: ReproductionOperator
    mutation: Optional[MutationOperator] = None
    integration: IntegrationOperator = field(
        default_factory=FirstThingsFirst
    )

    def start(self, callbacks: Optional[list[Callback]] = None) -> None:
        callbacks = [] if callbacks is None else callbacks

        parents = self.breeder.breed(self.init_size)
        self.fitness.evaluate(parents)
        for callback in callbacks:
            callback.on_start(parents)

        for _ in range(self.generations):

            for callback in callbacks:
                callback.on_generation_start(parents)

            offspring = self.reproduction.process(parents)
            if self.mutation is not None:
                offspring = self.mutation.process(offspring)
            self.fitness.evaluate(offspring)
            offspring, = self.integration.process((offspring, parents))
            parents = self.selection.process(offspring)

            for callback in callbacks:
                callback.on_generation_end(parents)

        for callback in callbacks:
            callback.on_end(parents)


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

    def _initialize_parameters(self, signature: str) -> None:
        # evaluates the given signature of the evolutionary strategy
        match = re.search(self.signature_regex, signature)
        if match is None:
            raise ValueError("Given signature does not match the "
                             "required pattern")
        matches = match.groups()
        self._ind_parent_selection: bool = matches[6] == "+"
        self._pop_parent_selection: bool = matches[2] == "+"
        # mu: number of parents
        self._ind_mu: int = int(matches[4])
        self._pop_mu: int = 1 if matches[0] is None else int(matches[0])
        # lambda: number of offspring
        ind_lambda: int = int(matches[7])
        pop_lambda: int = 1 if matches[3] is None else int(matches[3])
        # rho: mixin proportion number
        ind_rho: int = 1 if matches[5] is None else int(matches[5][1:])
        pop_rho: int = 1 if matches[1] is None else int(matches[1][1:])
        # gamma: cycle number for single population evolution
        self._ind_gamma: int = 1 if matches[8] is None else int(matches[8][1:])
        # operators
        self._selection = Best(
            in_size=(
                ind_lambda + self._ind_mu if self._ind_parent_selection
                else ind_lambda
            ),
            out_size=self._ind_mu,
        )
        self._pop_selection = BestMean(
            in_size=(
                pop_lambda + self._pop_mu if self._pop_parent_selection
                else pop_lambda
            ),
            out_size=self._pop_mu,
        )
        self._reproduction = MultiMix(in_size=ind_rho)
        self._reproduction.iter_type = NRandomBatchesIteration(
            batch_size=ind_rho,
            total=ind_lambda,
        )
        self._pop_reproduction = EquiMix(
            in_size=self._pop_mu,
            out_size=pop_lambda,
            group_size=pop_rho,
        )
        self._mutation = NormalDist(alpha=self.alpha, prob=1.0)

    def start(self, callbacks: Optional[list[Callback]] = None) -> None:
        callbacks = [] if callbacks is None else callbacks

        populations: list[Population] = []
        for i in range(self._pop_mu):
            populations.append(self.breeder.breed(self._ind_mu))
            if self._pop_parent_selection:
                self.fitness.evaluate(populations[-1])
            for callback in callbacks:
                callback.on_start(populations[-1])

        for _ in range(self.generations):
            offspring_populations = list(self._pop_reproduction.process(
                tuple(populations)
            ))

            for _ in range(self._ind_gamma):
                for i, population in enumerate(offspring_populations):
                    for callback in callbacks:
                        callback.on_generation_start(population)

                    offspring = self._reproduction.process(population)
                    offspring = self._mutation.process(offspring)
                    self.fitness.evaluate(offspring)
                    if self._ind_parent_selection:
                        offspring.populate(population)
                    offspring_populations[i] = self._selection.process(
                        offspring
                    )

                    for callback in callbacks:
                        callback.on_generation_end(offspring_populations[i])

            if self._pop_parent_selection:
                for population in populations:
                    offspring_populations.append(population)
            populations = list(
                self._pop_selection.process(tuple(offspring_populations))
            )
