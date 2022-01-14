import numpy as np

from peal.genetics import GPPool, GPTerminal
from peal.individual import Individual
from peal.operations.iteration import SingleIteration
from peal.operations.operator import Operator


class MutationOperator(Operator):
    """Operator for the mutation of individuals in a popoulation."""

    def __init__(
        self,
        in_size: int,
        out_size: int,
    ):
        super().__init__(
            in_size=in_size,
            out_size=out_size,
            iter_type=SingleIteration(),
        )


class BitFlip(MutationOperator):
    """Mutation that applies the python ``not`` operator to genes in
    a individual.

    Args:
        prob (float, optional): The probability of each gene to mutate.
            Defaults to 0.1.
    """

    def __init__(self, prob: float = 0.1):
        super().__init__(in_size=1, out_size=1)
        self._prob = prob

    def _process(
        self,
        individuals: tuple[Individual, ...],
    ) -> tuple[Individual, ...]:
        ind = individuals[0].copy()
        for i, gene in enumerate(ind.genes):
            if np.random.random_sample() <= self._prob:
                ind.genes[i] = not gene
        return (ind, )


class UniformInt(MutationOperator):
    """Mutation that selects a random uniformly distributed integer from
    a given range with a certain probability for a single gene.

    Args:
        prob (float, optional): The probability of each gene to mutate.
            Defaults to 0.1.
        lowest (int, optional): The lowest integer the mutation can turn
            a gene to. Defaults to -1.
        highest (int, optional): The highest integer the mutation can
            turn a gene to. Defaults to 1.
    """

    def __init__(
        self,
        prob: float = 0.1,
        lowest: int = -1,
        highest: int = 1,
    ):
        super().__init__(in_size=1, out_size=1)
        self._prob = prob
        self._lowest = lowest
        self._highest = highest

    def _process(
        self,
        individuals: tuple[Individual, ...],
    ) -> tuple[Individual, ...]:
        ind = individuals[0].copy()
        hits = np.where(
            np.random.random_sample(len(ind.genes)) <= self._prob
        )[0]
        ind.genes[hits] = np.random.randint(
            self._lowest,
            self._highest+1,
            size=len(hits),
        )
        return (ind, )


class NormalDist(MutationOperator):
    """Mutation operator that changes genes for an individual with a
    probability by __adding__ a randomly distributed real value.

    Args:
        prob (float, optional): The probability of each gene to mutate.
            Defaults to 0.1.
        mu (float, optional): The mean of the normal distribution the
            values are drawn from. Defaults to 0.
        sigma (float, optional): The standard deviation of the normal
            distribution the values are drawn from. Defaults to 1.
    """

    def __init__(
        self,
        prob: float = 0.1,
        mu: float = 0.0,
        sigma: float = 1.0,
    ):
        super().__init__(1, 1)
        self._prob = prob
        self._mu = mu
        self._sigma = sigma

    def _process(
        self,
        individuals: tuple[Individual, ...],
    ) -> tuple[Individual, ...]:
        ind = individuals[0].copy()
        hits = np.where(
            np.random.random_sample(len(ind.genes)) <= self._prob
        )[0]
        ind.genes[hits] += np.random.normal(
            self._mu,
            self._sigma,
            size=len(hits),
        )
        return (ind, )


class GPPoint(MutationOperator):
    """Point mutation used in a genetic programming algorithm.
    This mutation replaces a node in a genome tree by a subtree.

    Args:
        gene_pool (GPPool): The gene pool used to generate a genome
            tree for individuals.
        min_height (int, optional): The minimal height of the replacing
            subtree. Defaults to 1.
        max_height (int, optional): The maximal height of the replacing
            subtree. Defaults to 1.
        prob (float, optional): The probability to mutate one node in
            the tree representation of an individual. Defaults to 0.1.
    """

    def __init__(
        self,
        gene_pool: GPPool,
        min_height: int = 1,
        max_height: int = 1,
        prob: float = 0.1
    ):
        super().__init__(in_size=1, out_size=1)
        self._pool = gene_pool
        self._min_height = min_height
        self._max_height = max_height
        self._prob = prob

    def _process(
        self,
        individuals: tuple[Individual, ...],
    ) -> tuple[Individual, ...]:
        if np.random.random_sample() >= self._prob:
            return (individuals[0].copy(), )

        ind = individuals[0].copy()
        index = np.random.randint(0, len(ind.genes))
        # search for subtree slice starting at index in the tree
        right = index + 1
        total = 0
        if not isinstance(ind.genes[index], GPTerminal):
            total = len(ind.genes[index].argtypes)
        while total > 0:
            if isinstance(ind.genes[right], GPTerminal):
                total -= 1
            else:
                total -= len(ind.genes[right].argtypes) - 1
            right += 1
        ind.genes = np.concatenate((
            ind.genes[:index],
            self._pool.random_genome(
                rtype=ind.genes[index].rtype,
                height=np.random.randint(self._min_height, self._max_height+1),
            ),
            ind.genes[right:],
        ))
        return (ind, )
