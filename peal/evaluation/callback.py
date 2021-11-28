"""Module that implements the base callback class."""

import numpy as np
import numpy.typing as npt

from peal.population import Population


class Callback:
    """Base class for all callbacks that can be used in an evolutionary
    process.
    """

    def on_start(self, population: Population):
        """Will be called at the start of an evolutionary process."""

    def on_generation_start(self, population: Population):
        """Will be called at the start of each generation."""

    def on_generation_end(self, population: Population):
        """Will be called at the end of each generation."""

    def on_end(self, population: Population):
        """Will be called at the end of an evolutionary process."""


class BestWorstTracker(Callback):
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

    def on_start(self, population: Population):
        self.best = Population()
        self.worst = Population()

    def on_generation_end(self, population: Population):
        self.best.populate(max(population, key=lambda ind: ind.fitness))
        self.worst.populate(min(population, key=lambda ind: ind.fitness))


class DiversityStatistics(Callback):
    """A callback that computes the gene diversity in a population.

    Args:
        allele (np.ndarray): A numpy array containing all possible
            alleles each gene of an individual can have.

    Attributes:
        gene_diversity (np.ndarray): A numpy array containing the gene
            diversity for each generation at each locus.
        diversity (np.ndarray): The mean value of ``gene_diversity`` in
            each generation. Afterwards it will be scaled by
            ``allele.size/(allele.size-1)``.
    """

    def __init__(self, allele: npt.NDArray[float]):
        self._allele = allele
        self.gene_diversity: npt.NDArray[float] = np.empty((0, 1), dtype=float)

    @property
    def diversity(self) -> npt.NDArray[float]:
        """Scaled average gene diversity as a float between 0 and 1."""
        return (
            self._allele.size / (self._allele.size-1)
            * self.gene_diversity.mean(axis=1)
        )

    def on_start(self, population: Population):
        self.gene_diversity = np.zeros(
            (0, population[0].genes.shape[0]),
            dtype=float
        )

    def on_generation_end(self, population: Population):
        div: np.ndarray = np.ones((population[0].genes.shape[0],))
        for allele in self._allele:
            div -= (
                np.sum(population.genes == allele, axis=0) / population.size
            )**2
        self.gene_diversity = np.vstack([
            self.gene_diversity,
            div
        ])
