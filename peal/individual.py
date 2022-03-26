import numpy as np


class Individual:
    """Class for individuals used in an evolutionary algorithm.
    The genes of one individual represent a solution to an optimisation
    problem. The value or 'fitness' of the individual is a measure of
    the goodness-of-fit of this solution.

    Args:
        genes (np.ndarray): The genome of the individual as a numpy
            array containing all genes.
    """

    __slots__ = ("genes", "fitness", "hidden_genes")

    def __init__(self, genes: np.ndarray) -> None:
        self.genes = genes
        self.fitness = 0.0
        self.hidden_genes = np.ones((1,), dtype=np.float32)

    def copy(self) -> "Individual":
        """Creates and returns a copy of this individual."""
        ind = Individual(self.genes.copy())
        ind.fitness = self.fitness
        ind.hidden_genes = self.hidden_genes.copy()
        return ind

    def __repr__(self) -> str:
        return f"Individual(fitness={self.fitness}, genes={self.genes})"
