"""Module that provides operators for producing new populations out of
existing ones.
"""

from typing import Optional
import numpy as np

from peal.community import Community
from peal.genetics import GenePool
from peal.operators.iteration import StraightIteration
from peal.operators.operator import Operator
from peal.population import Population


class EquiMix(Operator):
    """Operator that mixes a number of populations to create a new ones.
    The number of individuals taken from each input population is the
    same. All populations are therefore expected to have the same size.

    Args:
        in_size (int): Number of populations to mix.
        out_size (int): Number of populations to create by mixing the
            input populations.
        group_size (int): Number of (randomly chosen) populations to
            mix for one new population. Each of these populations give
            the same number of individuals.
    """

    def __init__(self, in_size: int, out_size: int, group_size: int) -> None:
        super().__init__(StraightIteration(batch_size=in_size))
        self._group_size = group_size
        self._out_size = out_size

    def _process_community(
        self,
        container: Community,
        /, *,
        pool: Optional[GenePool] = None,
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
