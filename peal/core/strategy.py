from dataclasses import dataclass, field
import re
from typing import ClassVar
from peal.community import Community

from peal.operations.iteration import NRandomBatchesIteration
from peal.operations.mutation import NormalDist
from peal.operations.operator import Operator
from peal.operations.reproduction import CEquiMix, MultiMix, CCopy
from peal.operations.selection import CBestMean, Best
from peal.operations.integration import IntegrationOperator, FirstThingsFirst
from peal.population import Population


@dataclass
class Strategy:
    """Class that describes an evolutionary strategy. It can be executed
    within a :class:`~peal.core.environment.Environment`.

    Args:
        init_individuals (int): Number of individuals that are
            initialized at the start of evolution.
        generations (int): Number of generations to create, i.e. the
            number of times to iteratively execute this strategy.
        reproduction (Operator[Population]): The operator to use for the
            reproduction of individuals.
        mutation (Operator[Population]): The operator to use for the
            mutation of individuals.
        selection (Operator[Population]): The operator to use for the
            selection of individuals.
        integration (IntegrationOperator, optional): The operator to use
            for integrating the offspring population into the parent
            population in each step. Defaults to
            :class:`~peal.operations.integration.FirstThingsFirst`.
        init_populations (int, optional): The number of populations to
            start the evolution with, i.e. the size of the used
            :class:`~peal.community.Community`. Defaults to 1.
        population_generation (int, optional): While ``generations``
            describes the number of iterations evolving individuals in
            each population, this integer represents the number of times
            the whole community is evolved using the population
            operators. Defaults to 1.
        select_parent_populations (bool, optional): If true, the parent
            populations will be included in the selection process on
            community level. Defaults to true.
        population_selection (Operator[Community], optional): The
            operator that will be used to select populations from the
            community.
            Defaults to :class:`~peal.operations.reproduction.CCopy`.
        population_reproduction (Operator[Community], optional): The
            operator that will be used to create new populations out of
            the already existing ones in the community.
            Defaults to :class:`~peal.operations.reproduction.CCopy`.
    """

    SIGNATURE_RE: ClassVar[str] = (
        r"(?:(\d+)(/\d+)?(\+|,)(\d+))?\((\d+)(/\d+)?(\+|,)(\d+)\)(\^\d+)?"
    )

    init_individuals: int
    generations: int

    reproduction: Operator[Population]
    mutation: Operator[Population]
    selection: Operator[Population]

    integration: IntegrationOperator = field(
        default_factory=FirstThingsFirst,
    )

    init_populations: int = 1
    population_generations: int = 1
    select_parent_populations: bool = True

    population_selection: Operator[Community] = field(
        default_factory=CCopy,
    )
    population_reproduction: Operator[Community] = field(
        default_factory=CCopy,
    )

    @staticmethod
    def from_string(string: str, population_generations: int) -> "Strategy":
        """Creates an evolutionary strategy based on the notation
        Schwefel (1977) used to characterize multimembered evolutionary
        strategies. This is a string matching the expression
        ``a/b{,|+}c(d/e{,|+}f)^g`` where ``{,|+}`` denotes the
        character ``,`` or ``+``. Other symbols are used to describe the
        following.

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

        Args:
            string (str): A string matching the described expression.
            population_generations (int): An integer characterizing the
                number of iterations a community is evolved.
        """
        match = re.search(Strategy.SIGNATURE_RE, string)
        if match is None:
            raise ValueError("Given signature does not match the "
                             "required pattern")
        matches = match.groups()
        ind_parent_selection: bool = matches[6] == "+"
        pop_parent_selection: bool = matches[2] == "+"
        # mu: number of parents
        ind_mu: int = int(matches[4])
        pop_mu: int = 1 if matches[0] is None else int(matches[0])
        # lambda: number of offspring
        ind_lambda: int = int(matches[7])
        pop_lambda: int = 1 if matches[3] is None else int(matches[3])
        # rho: mixin proportion number
        ind_rho: int = 1 if matches[5] is None else int(matches[5][1:])
        pop_rho: int = 1 if matches[1] is None else int(matches[1][1:])
        # gamma: cycle number for single population evolution
        ind_gamma: int = 1 if matches[8] is None else int(matches[8][1:])
        # operators
        selection = Best(
            in_size=(
                ind_lambda + ind_mu if ind_parent_selection
                else ind_lambda
            ),
            out_size=ind_mu,
        )
        pop_selection = CBestMean(
            in_size=pop_lambda+pop_mu if pop_parent_selection else pop_lambda,
            out_size=pop_mu,
        )
        reproduction = MultiMix(in_size=ind_rho)
        reproduction.iter_type = NRandomBatchesIteration(
            batch_size=ind_rho,
            total=ind_lambda,
        )
        pop_reproduction = CEquiMix(
            in_size=pop_mu,
            out_size=pop_lambda,
            group_size=pop_rho,
        )
        mutation = NormalDist(alpha=1.3, prob=1.0)
        integration = FirstThingsFirst(
            size=ind_mu+ind_lambda if ind_parent_selection else ind_lambda
        )

        return Strategy(
            init_individuals=ind_mu,
            generations=ind_gamma,
            reproduction=reproduction,
            mutation=mutation,
            selection=selection,
            integration=integration,
            init_populations=pop_mu,
            population_generations=population_generations,
            select_parent_populations=pop_parent_selection,
            population_reproduction=pop_reproduction,
            population_selection=pop_selection,
        )
