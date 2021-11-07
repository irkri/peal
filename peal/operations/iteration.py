"""Module that defines certain iteration methods for populations."""

from typing import Iterator

import numpy as np

from peal.population import Individual, Population


def straight(
    population: Population,
    batch_size: int = 1,
) -> Iterator[list[Individual]]:
    for i in range(0, population.size, batch_size):
        yield population[i:i+batch_size]


def every(
    population: Population,
) -> Iterator[Individual]:
    for i in range(population.size):
        yield population[i]


def random_batches(
    population: Population,
    n: int = 1,
    size: int = 2,
) -> Iterator[list[Individual]]:
    for _ in range(n):
        indices = np.random.choice(population.size, size=size, replace=False)
        yield [population[i] for i in indices]
