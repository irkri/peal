"""Module that implements common mutation operations."""

import numpy as np

from peal.operations.mutation.base import MutationOperator
from peal.population import Population
from peal.operations.iteration import every


class BitFlip(MutationOperator):
    """Mutation that applies the python ``not`` operator to genes in
    a individual.

    Args:
        prob (float, optional): The probability of each gene to mutate.
            Defaults to 0.5.
    """

    def __init__(self, prob: float = 0.5):
        self._prob = prob

    def process(self, population: Population) -> Population:
        iterator = every(population)
        result = Population()
        for ind in iterator:
            new_ind = ind.copy()
            for i, gene in enumerate(ind.genes):
                if np.random.random_sample() <= self._prob:
                    new_ind.genes[i] = not gene
            result.populate(new_ind)
        return result


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
        self._prob = prob
        self._lowest = lowest
        self._highest = highest

    def process(self, population: Population) -> Population:
        iterator = every(population)
        result = Population()
        for ind in iterator:
            new_ind = ind.copy()
            hits = np.where(
                np.random.random_sample(len(ind.genes)) <= self._prob
            )[0]
            new_ind.genes[hits] = np.random.randint(
                self._lowest,
                self._highest+1,
                size=len(hits)
            )
            result.populate(new_ind)
        return result
