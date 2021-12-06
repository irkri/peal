"""Module that implements common mutation operations."""

from typing import Union

import numpy as np

from peal.operations.mutation.base import MutationOperator
from peal.population import Individual


class BitFlip(MutationOperator):
    """Mutation that applies the python ``not`` operator to genes in
    a individual.

    Args:
        prob (float, optional): The probability of each gene to mutate.
            Defaults to 0.5.
    """

    def __init__(self, prob: float = 0.5):
        super().__init__(1, 1)
        self._prob = prob

    def _process(
        self,
        individuals: Union[Individual, tuple[Individual, ...]],
    ) -> Union[Individual, tuple[Individual, ...]]:
        if not isinstance(individuals, Individual):
            raise TypeError("BitFlip expects a single individual")

        ind = individuals.copy()
        for i, gene in enumerate(ind.genes):
            if np.random.random_sample() <= self._prob:
                ind.genes[i] = not gene
        return ind


class UniformInt(MutationOperator):
    """Mutation that selects a random uniformly distributed integer from
    a given range with a certain probability for a single gene.

    Args:
        prob (float, optional): The probability of each gene to mutate.
            Defaults to 0.5.
        lowest (int, optional): The lowest integer the mutation can turn
            a gene to. Defaults to -1.
        highest (int, optional): The highest integer the mutation can
            turn a gene to. Defaults to 1.
    """

    def __init__(
        self,
        prob: float = 0.5,
        lowest: int = -1,
        highest: int = 1,
    ):
        super().__init__(1, 1)
        self._prob = prob
        self._lowest = lowest
        self._highest = highest

    def _process(
        self,
        individuals: Union[Individual, tuple[Individual, ...]],
    ) -> Union[Individual, tuple[Individual, ...]]:
        if not isinstance(individuals, Individual):
            raise TypeError("UniformInt expects a single individual")

        ind = individuals.copy()
        hits = np.where(
            np.random.random_sample(len(ind.genes)) <= self._prob
        )[0]
        ind.genes[hits] = np.random.randint(
            self._lowest,
            self._highest+1,
            size=len(hits)
        )
        return ind
