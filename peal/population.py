from typing import Optional, Union

import numpy as np

from peal.individual import Individual


class Population:
    """An iterable container for :class:`~peal.individual.Individual`
    objects.

    Args:
        individuals (Individual): One or more individuals to add.
    """

    def __init__(
        self,
        individuals: Optional[
            Union[Individual, tuple[Individual, ...], "Population"]
        ] = None,
    ):
        self._iter_id = -1
        self._individuals: list[Individual] = []
        if individuals is not None:
            self.populate(individuals)

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

    def populate(
        self,
        individuals: Union[Individual, tuple[Individual, ...], "Population"],
    ):
        """Add individuals to the population.

        Args:
            individuals (Individual | tuple | Population): One or
                a tuple of multiple individuals or a population of
                individuals to fill this population with.
        """
        if isinstance(individuals, Individual):
            self._individuals.append(individuals)
        elif isinstance(individuals, tuple):
            for individual in individuals:
                if not isinstance(individual, Individual):
                    raise TypeError("Can only append individuals to a "
                                    f"population, got {type(individual)}")
                self._individuals.append(individual)
        elif isinstance(individuals, Population):
            for individual in individuals:
                self._individuals.append(individual)
        else:
            raise TypeError("Can only append individuals to a population, "
                            f"got {type(individuals)}")

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
        copy = Population(self)
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
            return Population(tuple(self._individuals[key]))
        return self._individuals.__getitem__(key)

    def __setitem__(self, key, value):
        self._individuals.__setitem__(key, value)

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return f"Population(size={self.size})"
