"""Module that defines a population used in a
:class:`~peal.environment.environment.Environment`.
"""

from typing import Union

import numpy as np

from peal.population.individual import Individual


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

    @property
    def fitness(self) -> list[float]:
        """Returns the fitness of all individuals in the population as
        a list of floats.
        """
        return [ind.fitness for ind in self._individuals]

    @property
    def genes(self) -> np.ndarray:
        """Returns the genes of all individuals in the population as
        a multidimensional numpy array.
        """
        return np.array([ind.genes for ind in self._individuals])

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

    def summary(self, max_lines: int = 4) -> str:
        """Returns a summary of the population as a string.

        Args:
            max_lines (int, optional): Maximal number of lines the
                string spans over. This allows to display some
                Individuals from the population. Defaults to 4.
        """
        string = f"Population ({self.size} Individuals)\n"
        if self.size <= max_lines:
            for ind in self._individuals[:-1]:
                string += "  + " + str(ind) + "\n"
            string += "  + " + str(self._individuals[-1])
        else:
            for ind in self._individuals[:int(max_lines / 2)+(max_lines % 2)]:
                string += "  + " + str(ind) + "\n"
            string += "   ...\n"
            for ind in self._individuals[-int(max_lines / 2):-1]:
                string += "  + " + str(ind) + "\n"
            string += "  + " + str(self._individuals[-1])
        return string

    def copy(self) -> "Population":
        """Returns a copy of this population without copying the
        individuals."""
        copy = Population()
        copy.populate(self)
        return copy

    def __iter__(self) -> "Population":
        self._iter_id = -1
        return self

    def __next__(self) -> Individual:
        if self._iter_id == len(self._individuals) - 1:
            raise StopIteration
        self._iter_id += 1
        return self._individuals[self._iter_id]

    def __getitem__(self, key):
        if isinstance(key, slice):
            return Population(*self._individuals.__getitem__(key))
        return self._individuals.__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            return Population(*self._individuals.__setitem__(key, value))
        return self._individuals.__setitem__(key, value)

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return f"Population(size={self.size})"
