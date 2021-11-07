"""This module defines a basic individual used in a evolutionary
algorithm.
"""

import numpy as np


class Individual:
    """Class for individuals used in a evolutionary algorithm.

    For defining your own individual with a gene type different of
    ``numpy.ndarray``, define::

        MyIndividual: peal.BaseIndividual[mytype]
    """

    def __init__(self, genes: np.ndarray):
        self._genes = genes
        self._fitness: float = 0.0

    @property
    def fitness(self) -> float:
        """Fitness of the individual as a float."""
        return self._fitness

    @fitness.setter
    def fitness(self, fitness: float):
        if not isinstance(self._fitness, float):
            raise TypeError("fitness has to be a float value")
        self._fitness = fitness

    @property
    def genes(self) -> np.ndarray:
        """Genes of the individual."""
        return self._genes

    @genes.setter
    def genes(self, genes: np.ndarray):
        if self._genes.shape != genes.shape:
            raise ValueError(f"Expected shape {self._genes.shape} of genes")
        self._genes = genes

    def copy(self) -> "Individual":
        """Creates and returns a shallow copy of this individual."""
        return Individual(self._genes.copy())

    def __repr__(self) -> str:
        return f"Individual(fitness={self._fitness}, genes={self._genes})"