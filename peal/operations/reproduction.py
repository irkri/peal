from typing import Optional

import numpy as np
from peal.community import Community

from peal.operations.iteration import (
    IterationType,
    SingleIteration,
    RandomStraightIteration,
    StraightIteration
)
from peal.operations.operator import CommunityOperator, PopulationOperator
from peal.population import Population


class PopulationReproductionOperator(PopulationOperator):
    """Operator for the reproduction of individuals in a popoulation."""

    def __init__(
        self,
        in_size: int = 1,
        probability: float = 1.0,
        iter_type: Optional[IterationType] = None,
    ):
        if iter_type is None:
            iter_type = RandomStraightIteration(
                batch_size=in_size,
                probability=probability
            )
        super().__init__(iter_type=iter_type)


class Copy(PopulationReproductionOperator):
    """Simple reproduction operator that copies a single individual."""

    def __init__(self):
        super().__init__(iter_type=SingleIteration[Population]())

    def _process(
        self,
        container: Population,
    ) -> Population:
        return container.deepcopy()


class Crossover(PopulationReproductionOperator):
    """Crossover reproduction operator.

    Args:
        npoints (int, optional): The number of points to use for the
            gene split in the crossover operation. Defaults to 2.
        probability (float, optional): The probability of performing the
            crossover operation. Defaults to 0.5.
    """

    def __init__(self, npoints: int = 2, probability: float = 0.5):
        super().__init__(in_size=2, probability=probability)
        self._npoints = npoints

    def _process(
        self,
        container: Population,
    ) -> Population:
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


class MultiMix(PopulationReproductionOperator):
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
        super().__init__(in_size=in_size, probability=probability)

    def _process(
        self,
        container: Population,
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


class CEquiMix(CommunityOperator):
    """Operator that mixes a number of populations to create a new ones.
    The number of individuals taken from each input population is the
    same. All populations are therefore expected to have the same size.

    Args:
        out_size (int): Number of populations to create by mixing the
            input populations.
        group_size (int): Number of (randomly chosen) populations to
            mix for one new population. Each of these populations give
            the same number of individuals.
    """

    def __init__(self, in_size: int, out_size: int, group_size: int):
        super().__init__(StraightIteration[Community](batch_size=in_size))
        self._group_size = group_size
        self._out_size = out_size

    def _process(
        self,
        container: Community,
    ) -> Community:
        offspring_populations = Community()
        population_parent_indices = [
            np.random.randint(
                0,
                container.size,
                size=self._group_size,
            ) for _ in range(self._out_size)
        ]
        n_indivs = container[0].size
        for indices in population_parent_indices:
            new_population = Population()
            parts = [n_indivs // self._group_size
                     for _ in range(self._group_size)]
            for i in range(n_indivs % self._group_size):
                parts[i % self._group_size] += 1
            for i in indices:
                for j in parts:
                    new_population.integrate(container[i][0:j].deepcopy())
            offspring_populations.integrate(new_population)
        return offspring_populations


class CCopy(CommunityOperator):
    """Operator that returns a deep copy of the input population."""

    def __init__(self):
        super().__init__(SingleIteration[Community]())

    def _process(
        self,
        container: Community,
    ) -> Community:
        return container.deepcopy()
