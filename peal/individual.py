from dataclasses import dataclass, field

import numpy as np


@dataclass
class Individual:
    """Class for individuals used in an evolutionary process. The genes
    of one individual represent a solution to an optimisation problem.
    The value or 'fitness' of the individual is a measure of the
    goodness-of-fit of this solution.

    Args:
        genes (np.ndarray): The genome of the individual as a numpy
            array containing all genes.
    """

    genes: np.ndarray
    fitness: float = field(default=0.0, init=False, compare=True)
    hidden_genes: np.ndarray = field(
        default_factory=lambda: np.empty((0,)),
        init=False,
        repr=False,
    )

    def copy(self) -> "Individual":
        """Creates and returns a shallow copy of this individual."""
        ind = Individual(self.genes.copy())
        ind.fitness = self.fitness
        ind.hidden_genes = self.hidden_genes
        return ind
