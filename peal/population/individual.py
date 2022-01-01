import numpy as np


class Individual:
    """Class for individuals used in an evolutionary process. The genes
    of one individual represent a solution to an optimisation problem.
    The value or 'fitness' of the individual is a measure of the
    goodness-of-fit of this solution.

    Args:
        genes (np.ndarray): The genome of the individual as a numpy
            array containing all genes.
    """

    def __init__(self, genes: np.ndarray):
        self._genes = genes
        self._fitness: float = 0.0
        self.fitted: bool = False
        self._hidden_genes: np.ndarray = np.empty((0,))

    @property
    def fitness(self) -> float:
        """Fitness of the individual as a float."""
        return self._fitness

    @fitness.setter
    def fitness(self, fitness: float):
        self.fitted = True
        if not isinstance(self._fitness, float):
            raise TypeError("fitness has to be a float value")
        self._fitness = fitness

    @property
    def genes(self) -> np.ndarray:
        """Genes of the individual as a numpy array that represents the
        solution to an optimization problem.
        """
        self.fitted = False
        return self._genes

    @genes.setter
    def genes(self, genes: np.ndarray):
        self.fitted = False
        if self._genes.dtype != object and self._genes.shape != genes.shape:
            raise ValueError(f"Expected shape {self._genes.shape} of genes")
        self._genes = genes

    @property
    def hidden_genes(self) -> np.ndarray:
        """A number of genes that may be used internally by some
        methods in peal. Typically, one shouldn't access or try to
        interpret these numbers.
        """
        return self._hidden_genes

    @hidden_genes.setter
    def hidden_genes(self, hidden_genes: np.ndarray):
        self._hidden_genes = hidden_genes

    def copy(self) -> "Individual":
        """Creates and returns a shallow copy of this individual."""
        ind = Individual(self._genes.copy())
        ind.fitness = self._fitness
        ind._hidden_genes = self._hidden_genes
        return ind

    def __repr__(self) -> str:
        return f"Individual(fitness={self._fitness}, genes={self._genes})"
