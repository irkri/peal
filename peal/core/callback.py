import numpy as np

from peal.genetics import GenePool, GeneType
from peal.population import Population


class Callback:
    """Base class for all callbacks. A callback in peal is a tool that
    helps the user to supervise an evolutionary process. This way, not
    only the result of such a process can be viewed, but also other
    information that would be normally hidden and overwritten (e.g. at
    the end of generations).
    """

    def on_start(self, population: Population) -> None:
        """Will be called at the start of an evolutionary process."""

    def on_generation_start(self, population: Population) -> None:
        """Will be called at the start of each generation."""

    def on_generation_end(self, population: Population) -> None:
        """Will be called at the end of each generation."""

    def on_end(self, population: Population) -> None:
        """Will be called at the end of an evolutionary process."""


class BestWorst(Callback):
    """Class that tracks the best and worst individuals in an
    evolutionary process for each generation.

    Attributes:
        best (Population): A population containing the best individuals
            (based on fitness) from each generation. Populations are
            ordered containers and in this case the order depends on the
            generations already passed.
        worst (Population): A population containing the worst
            individuals from each generation.
    """

    def __init__(self):
        self.best: Population = Population()
        self.worst: Population = Population()

    def on_start(self, population: Population) -> None:
        self.best = Population()
        self.worst = Population()

    def on_generation_end(self, population: Population) -> None:
        self.best.populate(max(population, key=lambda ind: ind.fitness))
        self.worst.populate(min(population, key=lambda ind: ind.fitness))


class Diversity(Callback):
    """A callback that computes the gene diversity in a population.

    Args:
        pool (GenePool): A gene pool that is used in the evolutionary
            process to generate individuals.
    """

    def __init__(self, pool: GenePool):
        self._pool = pool
        if GeneType.CONST_SIZE not in self._pool.typing:
            raise ValueError("Diversity not available for genomes of "
                             "variable length")
        self.gene_diversity: np.ndarray = np.empty((0, 1), dtype=float)

    @property
    def diversity(self) -> np.ndarray:
        """For gene pools consisting of only categorical gene types,
        this property is the scaled average gene diversity as a float
        between 0 and 1 at each locus in a genome. For metric genomes,
        it is the mean of standard deviations across the genes of a
        population over multiple generations.
        """
        if GeneType.METRIC not in self._pool.typing:
            return (
                self._pool.size / (self._pool.size - 1)
                * self.gene_diversity.mean(axis=1)
            )
        return self.gene_diversity.mean(axis=1)

    def on_start(self, population: Population) -> None:
        self.gene_diversity = np.zeros(
            (0, population[0].genes.shape[0]),
            dtype=float
        )

    def on_generation_end(self, population: Population) -> None:
        div: np.ndarray = np.ones((population[0].genes.shape[0],))
        if GeneType.METRIC not in self._pool.typing:
            unique = set(np.hstack(list(population.genes.flatten())))
            for value in unique:
                div -= (
                    np.sum(population.genes == value, axis=0)
                    / population.size
                )**2
        else:
            div = np.zeros((population[0].genes.shape[0],))
            div += np.std(population.genes, axis=0)
        self.gene_diversity = np.vstack([
            self.gene_diversity,
            div
        ])
