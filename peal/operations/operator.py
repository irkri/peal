from abc import ABC, abstractmethod
from typing import Union

from peal.operations.iteration import IterationType
from peal.population import Individual, Population


class Operator(ABC):
    """Base class for evolutionary operators in peal.

    Args:
        in_individuals (int): The number of individuals to input in the
            operator.
        out_individuals (int): The number of individuals returned by the
            operator.
    """

    def __init__(
        self,
        in_individuals: int,
        out_individuals: int,
        iteration_type: IterationType,
    ):
        self._in_size = in_individuals
        self._out_size = out_individuals
        self._iteration_type = iteration_type

    def process(
        self,
        individuals: Union[Individual, tuple[Individual, ...], Population],
    ) -> Union[Individual, tuple[Individual, ...], Population]:
        """Processes the given individuals with the operator.
        The number of individuals to input and output is specified in
        the initialization of the operator.

        Args:
            individuals (Individual | tuple | Population): One or a
                tuple of more individuals to process. However, only one
                operation, that depends on one or more individuals, is
                performed. If a population is given, then the method
                :meth:`process_all` will be called instead.

        Returns:
            Processed individual(s) or population.
        """
        if isinstance(individuals, Population):
            return self.process_all(individuals)

        if (isinstance(individuals, Individual)
                and self._in_size != 1):
            raise ValueError("Too many individuals given to the operator, "
                             "expected 1")
        if (isinstance(individuals, tuple)
                and len(individuals) != self._in_size):
            raise ValueError("Incorrect number of individuals given to "
                             f"the operator: {len(individuals)}, "
                             f"expected {self._in_size}")
        out = self._process(individuals)
        if isinstance(out, tuple):
            if len(out) != self._out_size:
                raise ValueError("Incorrect number of individuals returned "
                                 f"by the operator: {len(out)}, "
                                 f"expected {self._out_size}")
        elif isinstance(out, Individual) and self._out_size != 1:
            raise ValueError("Not enough individuals returned by the operator")
        elif not isinstance(out, Individual):
            raise TypeError("Unknown type returned by the operator: "
                            f"{type(out)}")
        return out

    def process_all(
        self,
        population: Population,
    ) -> Population:
        """Processes all individuals in the given population and returns
        a new population. The result of this function depends on the
        specified iteration type in the initialization of this operator.
        """
        new_population = Population()
        iteration = self._iteration_type.iterate(population)
        for batch in iteration:
            new_population.populate(self._process(batch))
        return new_population

    @abstractmethod
    def _process(
        self,
        individuals: Union[Individual, tuple[Individual, ...]],
    ) -> Union[Individual, tuple[Individual, ...]]:
        ...

    def __call__(
        self,
        individuals: Union[Individual, tuple[Individual, ...], Population],
    ) -> Union[Individual, tuple[Individual, ...], Population]:
        return self.process(individuals)
