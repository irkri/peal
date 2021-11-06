"""Module that implements the base callback class."""

from peal.environment.population import Population


class Callback:
    """Base class for all callbacks used in an
    :class:`~peal.environment.base.Environment`.
    """

    def on_start(self, population: Population):
        """Will be called at the start of an evolutionary process."""

    def on_generation_start(self, population: Population):
        """Will be called at the start of each generation."""

    def on_selection(self, population: Population):
        """Will be called after all selection operations."""

    def on_reproduction(self, population: Population):
        """Will be called after all reproduction operations."""

    def on_mutation(self, population: Population):
        """Will be called after all mutation operations."""

    def on_generation_end(self, population: Population):
        """Will be called at the end of an evolutionary process."""

    def on_end(self, population: Population):
        """Will be called at the end of an evolutionary process."""


class FitnessTracker(Callback):
    """Class that tracks information on the fitness of individuals in
    the population.
    """

    def __init__(self):
        self.best: list[float] = []
        self.average: list[float] = []
        self.worst: list[float] = []

    def on_start(self, population: Population):
        self.best = []
        self.average = []
        self.worst = []

    def on_generation_start(self, population: Population):
        fitness = [indiv.fitness for indiv in population]
        self.best.append(max(fitness))
        self.average.append(sum(fitness)/population.size)
        self.worst.append(min(fitness))

    def on_end(self, population: Population):
        fitness = [indiv.fitness for indiv in population]
        self.best.append(max(fitness))
        self.average.append(sum(fitness)/population.size)
        self.worst.append(min(fitness))
