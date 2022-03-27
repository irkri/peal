"""Module that provides iteration types that are used in evolutionary
operators to iterate through populations or communities.
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional, Union, overload

import numpy as np

from peal.community import Community
from peal.population import Population


class IterationType(ABC):
    """Abstract class for an instruction on how to iterate over a
    population or community.
    """

    @abstractmethod
    def _iterate(
        self,
        container: Union[Population, Community],
    ) -> Iterator[Union[Population, Community]]:
        """Returns an iterator over the given population or community
        which yields smaller populations or communities.
        """

    @overload
    def __call__(
        self,
        container: Community,
    ) -> Iterator[Community]:
        ...

    @overload
    def __call__(
        self,
        container: Population,
    ) -> Iterator[Population]:
        ...

    def __call__(
        self,
        container: Union[Population, Community],
    ) -> Iterator[Union[Population, Community]]:
        return self._iterate(container)


class SingleIteration(IterationType):
    """Class that iterates over single elements in a container in the
    same order they appear.
    """

    def _iterate(
        self,
        container: Union[Population, Community],
    ) -> Iterator[Union[Population, Community]]:
        for i in range(container.size):
            yield container[i:i+1]


class RandomSingleIteration(IterationType):
    """Class that iterates over single individuals in a population the
    same order they appear.

    Args:
        probability (float): The probability that a single individual is
            returned.
    """

    def __init__(self, probability: float) -> None:
        self._probability = probability

    def _iterate(
        self,
        container: Union[Population, Community],
    ) -> Iterator[Union[Population, Community]]:
        for i in range(container.size):
            if np.random.random_sample() <= self._probability:
                yield container[i:i+1]


class StraightIteration(IterationType):
    """Class that performes a straight iteration over a population. That
    means that for a given batch size, in each iteration this number of
    individuals is returned. The individuals are located within the
    population in the same order they are returned.

    Args:
        batch_size (int) : The number of individuals to return at once.
    """

    def __init__(self, batch_size: int) -> None:
        self._batch_size = batch_size

    def _iterate(
        self,
        container: Union[Population, Community],
    ) -> Iterator[Union[Population, Community]]:
        for i in range(0, container.size, self._batch_size):
            yield container[i:i+self._batch_size]


class RandomStraightIteration(IterationType):
    """Class that performes the same straight iteration as
    :class:`StraightIteration` but returns each batch only with a
    certain probability.

    Args:
        batch_size (int): The number of individuals to return at once.
        probability (float): The probability that each batch is returned
            with.
    """

    def __init__(self, batch_size: int, probability: float) -> None:
        self._batch_size = batch_size
        self._probability = probability

    def _iterate(
        self,
        container: Union[Population, Community],
    ) -> Iterator[Union[Population, Community]]:
        for i in range(0, container.size, self._batch_size):
            if np.random.random_sample() <= self._probability:
                yield container[i:i+self._batch_size]


class NRandomBatchesIteration(IterationType):
    """Class that iterates over the given population by yielding a
    specified number of randomly selected batches of a given size.

    Args:
        batch_size (int, optional): Size of each batch returned by the
            iterator. Defaults to 1.
        total (int, optional): Number of random batches to return.
            If set to None, the number of returned batches is equal to
            the size of the input population. Defaults to None.
    """

    def __init__(
        self,
        batch_size: int = 1,
        total: Optional[int] = None,
    ) -> None:
        if not isinstance(batch_size, int) and batch_size <= 0:
            raise ValueError("batch_size has to be an integer > 0")
        self._batch_size = batch_size
        self._total = total

    def _iterate(
        self,
        container: Union[Population, Community],
    ) -> Iterator[Union[Population, Community]]:
        total = self._total if self._total is not None else container.size
        for _ in range(total):
            indices = np.random.choice(
                container.size,
                size=self._batch_size,
                replace=False,
            )
            yield container[list(indices)]
