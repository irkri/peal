"""Module that provides operators that reproduce individuals."""

from typing import Optional
import numpy as np

from peal.community import Community
from peal.genetics import GenePool
from peal.operators.iteration import (
    SingleIteration,
    RandomStraightIteration,
    StraightIteration,
)
from peal.operators.operator import Operator
from peal.population import Population


class Copy(Operator):
    """Simple reproduction operator that copies single individuals or
    populations.
    """

    def __init__(self) -> None:
        super().__init__(iter_type=SingleIteration())

    def _process_population(
        self,
        container: Population,
        /, *,
        pool: Optional[GenePool] = None,
    ) -> Population:
        return container.deepcopy()

    def _process_community(
        self,
        container: Community,
        /, *,
        pool: Optional[GenePool] = None,
    ) -> Community:
        return container.deepcopy()


class Crossover(Operator):
    """Crossover reproduction operator.

    Args:
        npoints (int, optional): The number of points to use for the
            gene split in the crossover operation. Defaults to 2.
        probability (float, optional): The probability of performing the
            crossover operation. Defaults to 0.5.
    """

    def __init__(self, npoints: int = 2, probability: float = 0.5) -> None:
        super().__init__(StraightIteration(batch_size=2))
        self._npoints = npoints
        self._probability = probability

    def _process_population(
        self,
        container: Population,
        /, *,
        pool: Optional[GenePool] = None,
    ) -> Population:
        if np.random.random() <= self._probability:
            return container.copy()
        ind1, ind2 = container
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
        return Population((off1, off2))


class DiscreteRecombination(Operator):
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

    def __init__(self, in_size: int = 2, probability: float = 0.5) -> None:
        super().__init__(
            RandomStraightIteration(
                batch_size=in_size,
                probability=probability,
            ),
        )

    def _process_population(
        self,
        container: Population,
        /, *,
        pool: Optional[GenePool] = None,
    ) -> Population:
        if container.size == 1:
            return container.deepcopy()

        parts = [
            container[0].genes.shape[0] // container.size
            for _ in range(container.size)
        ]
        missing = container[0].genes.shape[0] % container.size
        for i in range(missing):
            parts[i] += 1
        parts.insert(0, 0)

        genes = np.zeros_like(container[0].genes)
        shuffled_indices = np.arange(container[0].genes.shape[0])
        np.random.shuffle(shuffled_indices)
        for i in range(len(parts)-1):
            genes[shuffled_indices[parts[i]:parts[i]+parts[i+1]]] = (
                container[i].genes[
                    shuffled_indices[parts[i]:parts[i]+parts[i+1]]
                ]
            )
        new_ind = container[0].copy()
        new_ind.genes = genes
        return Population(new_ind)
