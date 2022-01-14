from abc import abstractmethod
from typing import Iterator, Optional

import numpy as np

from peal.individual import Individual
from peal.population import Population


class IterationType:
    """Abstract class for an instruction on how to iterate over a
    population.
    """

    def __init__(self, out_size: int):
        self._out_size = out_size

    @abstractmethod
    def iterate(
        self,
        population: Population,
    ) -> Iterator[tuple[Individual, ...]]:
        """Returns an iterator over the given population. The iterator
        contains individuals or tuples of individuals.
        """


class SingleIteration(IterationType):
    """Class that iterates over single individuals in a population the
    same order they appear.
    """

    def __init__(self):
        super().__init__(out_size=1)

    def iterate(
        self,
        population: Population,
    ) -> Iterator[tuple[Individual, ...]]:
        for i in range(population.size):
            yield (population[i], )


class RandomSingleIteration(IterationType):
    """Class that iterates over single individuals in a population the
    same order they appear.

    Args:
        probability (float): The probability that a single individual is
            returned.
    """

    def __init__(self, probability: float):
        super().__init__(out_size=1)
        self._probability = probability

    def iterate(
        self,
        population: Population,
    ) -> Iterator[tuple[Individual, ...]]:
        for i in range(population.size):
            yield (population[i], )


class StraightIteration(IterationType):
    """Class that performes a straight iteration over a population. That
    means that for a given batch size, in each iteration this number of
    individuals is returned. The individuals are located within the
    population in the same order they are returned.

    Args:
        batch_size (int) : The number of individuals to return at once.
    """

    def __init__(self, batch_size: int):
        super().__init__(out_size=batch_size)
        self._batch_size = batch_size

    def iterate(
        self,
        population: Population,
    ) -> Iterator[tuple[Individual, ...]]:
        for i in range(0, population.size, self._batch_size):
            yield tuple(population[i:i+self._batch_size])


class RandomStraightIteration(IterationType):
    """Class that performes the same straight iteration as
    :class:`StraightIteration` but returns each batch only with a
    certain probability.

    Args:
        batch_size (int): The number of individuals to return at once.
        probability (float): The probability that each batch is returned
            with.
    """

    def __init__(self, batch_size: int, probability: float):
        super().__init__(out_size=batch_size)
        self._batch_size = batch_size
        self._probability = probability

    def iterate(
        self,
        population: Population,
    ) -> Iterator[tuple[Individual, ...]]:
        for i in range(0, population.size, self._batch_size):
            if np.random.random_sample() <= self._probability:
                yield tuple(population[i:i+self._batch_size])


class NRandomBatchesIteration(IterationType):
    """Class that iterates over the given population by yielding a
    specified number of randomly selected batches of a given size.

    Args:
        size (int, optional): Size of each batch returned by the
            iterator. Defaults to 1.
        total (int, optional): Number of random batches to return.
            If set to None, the number of returned batches is equal to
            the size of the input population. Defaults to None.
    """

    def __init__(self, batch_size: int = 1, total: Optional[int] = None):
        super().__init__(out_size=batch_size)
        self._batch_size = batch_size
        self._total = total

    def iterate(
        self,
        population: Population,
    ) -> Iterator[tuple[Individual, ...]]:
        total = self._total if self._total is not None else population.size
        for _ in range(total):
            indices = np.random.choice(
                population.size,
                size=self._batch_size,
                replace=False
            )
            yield tuple(population[i] for i in indices)
