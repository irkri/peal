"""Module that defines certain iteration methods for populations."""

from typing import Iterator

import numpy as np

from peal.population import Individual, Population


def straight(
    population: Population,
    batch_size: int = 1,
) -> Iterator[list[Individual]]:
    """Returns an iterator over the given population that returns
    a number of individuals at a time.

    Args:
        population (Population): Population to iterate over.
        batch_size (int, optional): Number of individuals returned in
            one batch. Defaults to 1.
    """
    for i in range(0, population.size, batch_size):
        yield population[i:i+batch_size]


def every(
    population: Population,
) -> Iterator[Individual]:
    """Returns an iterator over every individual in the given
    population.

    Args:
        population (Population): Population to iterate over.
    """
    for i in range(population.size):
        yield population[i]


def random_batches(
    population: Population,
    total: int = 1,
    size: int = 2,
) -> Iterator[list[Individual]]:
    """Returns an iterator over the given population that returns a
    number of randomly selected batches of a given size.

    Args:
        population (Population): Population to iterate over.
        total (int, optional): Number of random batches to return.
            Defaults to 1.
        size (int, optional): Size of each batch returned by the
            iterator.
    """
    for _ in range(total):
        indices = np.random.choice(population.size, size=size, replace=False)
        yield [population[i] for i in indices]
