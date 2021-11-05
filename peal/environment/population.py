"""Module that defines a population used in a
:class:`~peal.environment.environment.Environment`.
"""

from typing import Union

from peal.individual.base import Individual


class Population:
    """A iterable container for
    :class:`~peal.individual.base.Individual` objects.

    Args:
        individuals (Individual): One or more individuals to add.
    """

    def __init__(self, *individuals: Individual):
        self._iter_id = -1
        self._individuals: list[Individual] = []
        self.populate(*individuals)

    @property
    def size(self) -> int:
        """Returns the number of individuals in the population."""
        return len(self._individuals)

    def populate(self, *individuals: Union[Individual, "Population"]):
        """Add individuals to the population.

        Args:
            individuals (Individual | Population): One or more
                individuals to add or populations to add individuals
                from.
        """
        for individual in individuals:
            if isinstance(individual, Individual):
                self._individuals.append(individual)
            elif isinstance(individual, Population):
                for i in range(individual.size):
                    self._individuals.append(individual[i])
            else:
                raise TypeError("Can only append individuals to a population")

    def __iter__(self) -> "Population":
        self._iter_id = -1
        return self

    def __next__(self) -> Individual:
        if self._iter_id == len(self._individuals) - 1:
            raise StopIteration
        self._iter_id += 1
        return self._individuals[self._iter_id]

    def __getitem__(self, key):
        return self._individuals.__getitem__(key)
