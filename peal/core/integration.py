"""Module that implements some integration techniques used in
evolutionary processes.
"""

from abc import ABC, abstractmethod

import numpy as np

from peal.population import Population


class IntegrationTechnique(ABC):
    """Abstract class for an integration technique that is responsible
    to merge a given offspring population into the parent population.
    """

    @abstractmethod
    def merge(
        self,
        offspring: Population,
        parents: Population,
    ) -> Population:
        """Merges offspring and parent population into one."""


class OffspringFirstIntegration(IntegrationTechnique):
    """This integration technique prefers the offspring to put in the
    new population."""

    def merge(
        self,
        offspring: Population,
        parents: Population,
    ) -> Population:
        """Merges the individuals of offspring and parents into a new
        population of same size as the parent population. This technique
        prefers the offspring. If individuals are missing after the
        offspring merged, parents will be drawn (in the order they
        appear in ``parents``) until the desired size is reached.

        Args:
            offspring (Population): The offspring population.
            parent (Population): The parent population.

        Returns:
            Population: A population of same size as ``parents``.
        """
        if offspring.size >= parents.size:
            return offspring[:parents.size]

        merged = offspring.copy()
        for i in range(parents.size-offspring.size):
            merged.populate(parents[i])

        return merged


class CrowdedIntegration(IntegrationTechnique):
    """This integration technique implements the so called 'crowding' to
    an evolutionary process.

    Args:
        crowding_factor (int, optinoal): The number of parents to
            compare one child to. Known as the crowding factor.
            Defaults to 10.
    """

    def __init__(self, crowding_factor: int = 10):
        self._cf = crowding_factor

    def merge(
        self,
        offspring: Population,
        parents: Population,
    ) -> Population:
        """Merges the individuals of offspring and parents into a new
        population of same size as the parent population. This technique
        prefers the offspring. If individuals are missing after the
        offspring merged, parents will be drawn (in the order they
        appear in ``parents``) until the desired size is reached.

        Args:
            offspring (Population): The offspring population.
            parent (Population): The parent population.

        Returns:
            Population: A population of same size as ``parents``.
        """
        merged = parents.copy()

        for off in offspring:
            compare = np.random.choice(
                parents.size,
                size=self._cf,
                replace=False
            )
            best = compare[0]
            best_distance = np.sum(parents[compare[0]].genes == off.genes)
            for index in compare:
                if ((dist := np.sum(parents[index].genes == off.genes))
                        < best_distance):
                    best_distance = dist
                    best = index
            merged[best] = off.copy()
        return merged
