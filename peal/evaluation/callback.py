"""Module that implements the base callback class."""

from peal.population import Population


class Callback:
    """Base class for all callbacks that can be used in evolutionary
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
