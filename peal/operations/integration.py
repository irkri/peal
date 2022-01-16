import numpy as np

from peal.community import Community
from peal.operations.iteration import StraightIteration
from peal.operations.operator import CommunityOperator


class IntegrationOperator(CommunityOperator):
    """Abstract class for an integration operation that is responsible
    to merge a given offspring and parent population into one.
    """

    def __init__(self):
        super().__init__(StraightIteration[Community](batch_size=2))


class FirstThingsFirst(IntegrationOperator):
    """This integration operation merges the individuals of offspring
    and parents into a new population of same size as the parent
    population. At first, individuals from the first population supplied
    are taken.
    If individuals are missing after this population (either offspring
    or parents) merged, individuals from the other population are taken
    in the order they appear until the requested size is reached.

    Args:
        size (int): The size of the output population. There are two
            special cases. If ``size=-1`` then the output population
            will have the same size as the second population that is
            given to the operator, if ``size=-2`` the first populations
            size will be taken. For every other positive integer this
            argument is taken literally. Defaults to -1.
    """

    def __init__(self, size: int = -1):
        super().__init__()
        if not isinstance(size, int) or size < -2 or size == 0:
            raise ValueError("Attribute size has to be -2, -1 or a non-zero "
                             "positive integer")
        self._size = size

    def _process(
        self,
        container: Community,
    ) -> Community:
        pop1, pop2 = container
        req_size = self._size
        if self._size == -1:
            req_size = pop2.size
        elif self._size == -2:
            req_size = pop1.size
        elif self._size > pop1.size + pop2.size:
            raise RuntimeError("Given populations are too small for "
                               "requested size")

        if pop1.size >= req_size:
            return Community(pop1[:req_size])
        merged = pop1.deepcopy()
        merged.integrate(pop2[:req_size-pop1.size].deepcopy())

        return Community(merged)


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
        container: Community,
    ) -> Community:
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
        offspring, parents = container
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
        return Community(merged)
