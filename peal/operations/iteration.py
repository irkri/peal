from abc import ABC, abstractmethod
from typing import Generic, Iterator, Optional, TypeVar

import numpy as np

from peal.community import Community
from peal.population import Population

T_iteration = TypeVar("T_iteration", Population, Community)


class IterationType(ABC, Generic[T_iteration]):
    """Abstract class for an instruction on how to iterate over a
    population or community.
    """

    @abstractmethod
    def iterate(
        self,
        container: T_iteration,
    ) -> Iterator[T_iteration]:
        """Returns an iterator over the given population or community
        which yields smaller populations or communities.
        """


class SingleIteration(IterationType[T_iteration]):
    """Class that iterates over single elements in a container in the
    same order they appear.
    """

    def iterate(
        self,
        container: T_iteration,
    ) -> Iterator[T_iteration]:
        for i in range(container.size):
            yield container[i:i+1]


class RandomSingleIteration(IterationType[T_iteration]):
    """Class that iterates over single individuals in a population the
    same order they appear.

    Args:
        probability (float): The probability that a single individual is
            returned.
    """

    def __init__(self, probability: float):
        self._probability = probability

    def iterate(
        self,
        container: T_iteration,
    ) -> Iterator[T_iteration]:
        for i in range(container.size):
            if np.random.random_sample() <= self._probability:
                yield container[i:i+1]


class StraightIteration(IterationType[T_iteration]):
    """Class that performes a straight iteration over a population. That
    means that for a given batch size, in each iteration this number of
    individuals is returned. The individuals are located within the
    population in the same order they are returned.

    Args:
        batch_size (int) : The number of individuals to return at once.
    """

    def __init__(self, batch_size: int):
        self._batch_size = batch_size

    def iterate(
        self,
        container: T_iteration,
    ) -> Iterator[T_iteration]:
        for i in range(0, container.size, self._batch_size):
            yield container[i:i+self._batch_size]


class RandomStraightIteration(IterationType[T_iteration]):
    """Class that performes the same straight iteration as
    :class:`StraightIteration` but returns each batch only with a
    certain probability.

    Args:
        batch_size (int): The number of individuals to return at once.
        probability (float): The probability that each batch is returned
            with.
    """

    def __init__(self, batch_size: int, probability: float):
        self._batch_size = batch_size
        self._probability = probability

    def iterate(
        self,
        container: T_iteration,
    ) -> Iterator[T_iteration]:
        for i in range(0, container.size, self._batch_size):
            if np.random.random_sample() <= self._probability:
                yield container[i:i+self._batch_size]


class NRandomBatchesIteration(IterationType[T_iteration]):
    """Class that iterates over the given population by yielding a
    specified number of randomly selected batches of a given size.

    Args:
        batch_size (int, optional): Size of each batch returned by the
            iterator. Defaults to 1.
        total (int, optional): Number of random batches to return.
            If set to None, the number of returned batches is equal to
            the size of the input population. Defaults to None.
    """

    def __init__(self, batch_size: int = 1, total: Optional[int] = None):
        if not isinstance(batch_size, int) and batch_size <= 0:
            raise ValueError("batch_size has to be an integer > 0")
        self._batch_size = batch_size
        self._total = total

    def iterate(
        self,
        container: T_iteration,
    ) -> Iterator[T_iteration]:
        total = self._total if self._total is not None else container.size
        for _ in range(total):
            indices = np.random.choice(
                container.size,
                size=self._batch_size,
                replace=False,
            )
            yield container[list(indices)]
