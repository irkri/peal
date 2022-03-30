import re
from typing import ClassVar

from peal.operators.clash import EquiMix
from peal.operators.iteration import NRandomBatchesIteration
from peal.operators.mutation import NormalDist
from peal.operators.operator import OperatorChain
from peal.operators.reproduction import DiscreteRecombination
from peal.operators.selection import Best, BestMean


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
        r"(\[)?"
        r"(?:(\d+)(/\d+)?(\+|,)(\d+))?"
        r"\((\d+)(/\d+)?(\+|,)(\d+)\)(\^\d+)?"
        r"(?(1)\](\^\d+)?|$)"
    )

    def __init__(
        self,
        operator_chain: OperatorChain,
        /, *,
        init_size: int = 100,
        generations: int = 100,
    ) -> None:
        self.operator_chain = operator_chain
        self.init_size = init_size
        self.generations = generations

    @staticmethod
    def from_string(string: str, /) -> tuple["Strategy", "Strategy"]:
        """Creates an evolutionary strategies based on the notation
        Schwefel (1977) used to characterize multimembered evolutionary
        strategies. This is a string matching the expression
        ``[a/b{,|+}c(d/e{,|+}f)^g]^h`` where ``{,|+}`` denotes the
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
        * ``h``: Number of generations between population evolution.
            There are two for-loops in a strategy like this. The outer
            iterates ``h`` times, the inner ``g`` times.

        The values ``a``, ``b`` and ``c`` are optional, ``a`` and
        ``c`` always have to be supplied if one of them is
        specified. Also ``b``, ``e``, ``g`` and ``h`` are optional. But
        if they are given, the string also needs the corresponding
        enclosing brackets.
        Specifying ``d+f`` refers to the strategy of using also
        the parent individuals of one population for selecting the
        offspring while ``d,f`` only selects individuals from the
        mutated and reproduced original population. The same applies
        for ``a+c`` and ``a,c``, only on population level.

        Args:
            string (str): A string matching the described expression.
        """
        match = re.match(Strategy.SIGNATURE_RE, string)
        if match is None:
            raise ValueError("Given signature does not match the "
                             "required pattern")
        # ('[', 'a', '/b', ',+', 'c', 'd', '/e', ',+', 'f', '^g', '^h')
        matches = match.groups()
        ind_parent_selection: bool = matches[7] == "+"
        pop_parent_selection: bool = matches[3] == "+"
        # mu: number of parents
        ind_mu: int = int(matches[5])
        pop_mu: int = 1 if matches[1] is None else int(matches[1])
        # lambda: number of offspring
        ind_lambda: int = int(matches[8])
        pop_lambda: int = 1 if matches[4] is None else int(matches[4])
        # rho: mixin proportion number
        ind_rho: int = 1 if matches[6] is None else int(matches[6][1:])
        pop_rho: int = 1 if matches[2] is None else int(matches[2][1:])
        # gamma: number of generations
        ind_gamma: int = 1 if matches[9] is None else int(matches[9][1:])
        pop_gamma: int = 1 if matches[10] is None else int(matches[10][1:])
        # operators
        selection = Best(
            in_size=(
                ind_lambda + ind_mu if ind_parent_selection
                else ind_lambda
            ),
            out_size=ind_mu,
        )
        pop_selection = BestMean(
            in_size=pop_lambda+pop_mu if pop_parent_selection else pop_lambda,
            out_size=pop_mu,
        )
        reproduction = DiscreteRecombination(in_size=ind_rho)
        reproduction.iter_type = NRandomBatchesIteration(
            batch_size=ind_rho,
            total=ind_lambda,
        )
        pop_reproduction = EquiMix(
            in_size=pop_mu,
            out_size=pop_lambda,
            group_size=pop_rho,
        )
        mutation = NormalDist(alpha=1.3, prob=1.0)

        return (
            Strategy(
                OperatorChain(selection, reproduction, mutation),
                init_size=ind_mu,
                generations=ind_gamma,
            ),
            Strategy(
                OperatorChain(pop_selection, pop_reproduction),
                init_size=pop_mu,
                generations=pop_gamma,
            ),
       )
