"""Module that implement the base class for an evolutionary operator."""

from abc import ABC, abstractmethod

from peal.population.population import Population


class Operator(ABC):
    """Base class for evolutionary operators in peal."""

    @abstractmethod
    def process(self, population: Population) -> Population:
        """Processes a given population with the operator."""

    def __call__(self, population: Population) -> Population:
        return self.process(population)
