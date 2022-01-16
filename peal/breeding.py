from peal.genetics import GenePool
from peal.individual import Individual
from peal.population import Population


class Breeder:
    """Class that manages the generation of new
    :class:`~peal.individual.Individual` instances.

    Args:
        gene_pool (GenePool): The gene pool of
            :class:`~peal.individual.Individual` objects that can be
            created by this breeder.
    """

    def __init__(self, gene_pool: GenePool):
        self._gene_pool = gene_pool

    def breed(self, size: int = 1) -> Population:
        """Returns a population of given size.

        Args:
            size (int, optional): Number of individuals to breed.
                Defaults to 1.
        """
        population = Population()
        for _ in range(size):
            population.integrate(Individual(self._gene_pool.random_genome()))
        return population

    def __call__(self, size: int = 1) -> Population:
        return self.breed(size)
