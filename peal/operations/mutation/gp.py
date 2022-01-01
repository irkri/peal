from typing import Union

import numpy as np

from peal.operations.mutation.base import MutationOperator
from peal.population import Individual, GenePool


class GPPoint(MutationOperator):
    """Point mutation used in a genetic programming algorithm.
    This mutation replaces a node

    Args:
        gene_pool (GenePool): The gene pool used to generate a
        prob (float, optional): The probability to mutate one node in
            the tree representation of an individual. Defaults to 0.1.
    """

    def __init__(self, gene_pool: GenePool, prob: float = 0.1):
        super().__init__(1, 1)
        self._gene_pool = gene_pool
        self._prob = prob

    def _process(
        self,
        individuals: Union[Individual, tuple[Individual, ...]],
    ) -> Union[Individual, tuple[Individual, ...]]:
        if not isinstance(individuals, Individual):
            raise TypeError("GPPoint expects a single individual")
        if np.random.random_sample() >= self._prob:
            return individuals.copy()

        ind = individuals.copy()
        index = np.random.randint(0, len(ind.genes))
        # search for subtree slice starting at index in the tree
        right = index + 1
        total = len(ind.genes[index].argtypes)
        while total > 0:
            total += len(ind.genes[right].argtypes) - 1
            right += 1
        ind.genes = np.concatenate((
            ind.genes[:index],
            self._gene_pool.random_genome(rtype=ind.genes[index].rtype),
            ind.genes[right:],
        ))
        return ind
