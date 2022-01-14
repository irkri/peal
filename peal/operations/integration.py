import numpy as np

from peal.operations.operator import PopulationOperator

from peal.population import Population


class IntegrationOperator(PopulationOperator):
    """Abstract class for an integration operation that is responsible
    to merge a given offspring and parent population into one.
    """

    def __init__(self):
        super().__init__(in_size=2, out_size=1)


class OffspringFirst(IntegrationOperator):
    """This integration operation merges the individuals of offspring
    and parents into a new population of same size as the parent
    population. At first, individuals from the offspring population are
    taken.
    If individuals are missing after (part of) the offspring merged,
    parents will be drawn (in the order they appear in ``parents``)
    until the desired size is reached.
    """

    def _process(
        self,
        populations: tuple[Population, ...],
    ) -> tuple[Population, ...]:
        if not isinstance(populations, tuple):
            raise TypeError("Operator got unexpected type, has to be a tuple")

        offspring, parents = populations
        if offspring.size >= parents.size:
            return (offspring[:parents.size], )

        merged = offspring.copy()
        for i in range(parents.size-offspring.size):
            merged.populate(parents[i])

        return (merged, )


class Crowded(IntegrationOperator):
    """This integration technique implements the so called 'crowding' to
    an evolutionary process.

    Args:
        crowding_factor (int): The number of parents to
            compare one child to. Known as the crowding factor.
    """

    def __init__(self, crowding_factor: int):
        super().__init__()
        self._cf = crowding_factor

    def _process(
        self,
        populations: tuple[Population, ...],
    ) -> tuple[Population, ...]:
        """Merges the given offspring and parent population. Each
        offspring will be compared to a number of random individuals
        from the parent population corresponding to the initialized
        crowding factor. The parent that has the most similarities
        (i.e. smallest hamming distance) will be replaced by this
        offspring.

        Args:
            offspring (Population): The offspring population.
            parent (Population): The parent population.

        Returns:
            Population: A population of same size as ``parents``.
        """
        offspring, parents = populations
        merged = parents.copy()

        for off in offspring:
            compare = np.random.choice(
                parents.size,
                size=self._cf,
                replace=False
            )
            best = compare[0]
            best_distance = np.sum(parents[compare[0]].genes == off.genes)
            for index in compare:
                if ((dist := np.sum(parents[index].genes == off.genes))
                        > best_distance):
                    best_distance = dist
                    best = index
            merged[best] = off.copy()
        return (merged, )
