from typing import Optional

import numpy as np

from peal.operations.iteration import (
    IterationType,
    SingleIteration,
    RandomStraightIteration
)
from peal.operations.operator import Operator
from peal.population import Individual


class ReproductionOperator(Operator):
    """Operator for the reproduction of individuals in a popoulation."""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        probability: float = 1.0,
        iter_type: Optional[IterationType] = None,
    ):
        if iter_type is None:
            iter_type = RandomStraightIteration(
                batch_size=in_size,
                probability=probability
            )
        super().__init__(
            in_size=in_size,
            out_size=out_size,
            iter_type=iter_type,
        )


class Copy(ReproductionOperator):
    """Simple reproduction operator that copies a single individual."""

    def __init__(self):
        super().__init__(in_size=1, out_size=1, iter_type=SingleIteration())

    def _process(
        self,
        individuals: tuple[Individual, ...],
    ) -> tuple[Individual, ...]:
        return (individuals[0].copy(), )


class Crossover(ReproductionOperator):
    """Crossover reproduction operator.

    Args:
        npoints (int, optional): The number of points to use for the
            gene split in the crossover operation. Defaults to 2.
        probability (float, optional): The probability of performing the
            crossover operation. Defaults to 0.5.
    """

    def __init__(self, npoints: int = 2, probability: float = 0.5):
        super().__init__(in_size=2, out_size=2, probability=probability)
        self._npoints = npoints

    def _process(
        self,
        individuals: tuple[Individual, ...],
    ) -> tuple[Individual, ...]:
        ind1, ind2 = individuals
        points = np.insert(
            np.sort(
                np.random.randint(
                    1,
                    len(ind1.genes),
                    size=self._npoints
                )
            ),
            [0, -1],
            [0, len(ind1.genes)]
        )
        start = self._npoints % 2
        off1, off2 = ind1.copy(), ind2.copy()
        for i in range(start, self._npoints+(1-start), 2):
            off1.genes[points[i]:points[i+1]] = ind2.genes[
                points[i]:points[i+1]
            ]
            off2.genes[points[i]:points[i+1]] = ind1.genes[
                points[i]:points[i+1]
            ]
        return off1, off2


class MultiMix(ReproductionOperator):
    """Reproduction operator that mixes the genes of multiple
    individuals to create a new individual. Each individual gives the
    same proportion of their genes.
    This only works as intuitively explained if the number of input
    individuals doesn't exceed the number of genes an individual has and
    if each individual has the same number of genes.

    Args:
        in_size (int, optional): The number of input individuals to mix
            for the one output individual. Defaults to 2.
        probability (float, optional): The probability of performing the
            crossover operation. Defaults to 0.5.
    """

    def __init__(self, in_size: int = 2, probability: float = 0.5):
        super().__init__(in_size=in_size, out_size=1, probability=probability)

    def _process(
        self,
        individuals: tuple[Individual, ...],
    ) -> tuple[Individual, ...]:
        if self._in_size == 1:
            return (individuals[0].copy(), )

        parts = [
            individuals[0].genes.shape[0] // self._in_size
            for _ in range(self._in_size)
        ]
        missing = individuals[0].genes.shape[0] % self._in_size
        for i in range(missing):
            parts[i] += 1
        parts.insert(0, 0)

        genes = np.zeros_like(individuals[0].genes)
        shuffled_indices = np.arange(individuals[0].genes.shape[0])
        np.random.shuffle(shuffled_indices)
        for i in range(len(parts)-1):
            genes[shuffled_indices[parts[i]:parts[i]+parts[i+1]]] = (
                individuals[i].genes[
                    shuffled_indices[parts[i]:parts[i]+parts[i+1]]
                ]
            )
        new_ind = Individual(genes)
        return (new_ind, )
