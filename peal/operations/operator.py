from abc import ABC, abstractmethod
from typing import Optional, Union, overload

from peal.individual import Individual
from peal.operations.iteration import IterationType, SingleIteration
from peal.population import Population


class Operator(ABC):
    """Base class for evolutionary operators in peal.

    Args:
        in_size (int): The number of individuals to input in the
            operator.
        out_size (int, optional): The number of individuals returned by
            the operator. Defaults to ``in_size``.
        iter_type (IterationType, optional): The iteration type of the
            iterator. Defaults to
            :class:`~peal.operations.iteration.SingleIteration`.
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        iter_type: Optional[IterationType] = None,
    ):
        self._in_size = in_size
        self._out_size = out_size
        self._iter_type = SingleIteration() if iter_type is None else iter_type

    @property
    def iter_type(self) -> IterationType:
        """The type of iteration through a population to use if the
        operators :meth:`process` method is called on a population. This
        leads to the processing of single individuals in the population.
        In which order they are processed or how many individuals are
        processed at once will be specified by such an iteration type.
        """
        return self._iter_type

    @iter_type.setter
    def iter_type(self, iter_type: IterationType) -> None:
        if not isinstance(iter_type, IterationType):
            raise TypeError(f"Expected IterationType, got {type(iter_type)}")
        self._iter_type = iter_type

    @overload
    def process(self, individuals: Population) -> Population:
        ...

    @overload
    def process(
        self,
        individuals: tuple[Individual, ...]
    ) -> tuple[Individual, ...]:
        ...

    def process(
        self,
        individuals: Union[tuple[Individual, ...], Population],
    ) -> Union[tuple[Individual, ...], Population]:
        """Processes the given individuals or population with the
        operator. The number of objects to input and the size of the
        output is specified in the initialization of the operator.

        Args:
            objects (tuple[Individual, ...] | Population): A tuple of
                one or multiple individuals to process. Only one
                operation is performed. If a population is given then
                :meth:`process_all` will be called instead.

        Returns:
            Processed individual(s) or population.
        """
        if isinstance(individuals, Population):
            return self.process_all(individuals)

        if not isinstance(individuals, tuple):
            raise TypeError("Argument of operator has unknown type "
                            f"{type(individuals)}")
        if len(individuals) != self._in_size:
            raise ValueError("Wrong number of arguments to operator "
                             f"({len(individuals)}), "
                             f"should be {self._in_size}.")

        result = self._process(individuals)
        if not isinstance(result, tuple):
            raise TypeError("Return of operator has unknown type "
                            f"{type(individuals)}")
        elif len(result) != self._out_size:
            raise ValueError("Wrong number of values returned by operator "
                             f"({len(result)}), should be {self._out_size}.")
        return result

    def process_all(
        self,
        population: Population,
    ) -> Population:
        """Processes all individuals in the given population and returns
        a new population. The result of this function depends on the
        specified iteration type in the initialization of this operator.
        """
        new_population = Population()
        iteration = self._iter_type.iterate(population)
        for batch in iteration:
            new_population.populate(self._process(batch))
        return new_population

    @abstractmethod
    def _process(
        self,
        individuals: tuple[Individual, ...],
    ) -> tuple[Individual, ...]:
        ...


class PopulationOperator(ABC):
    """Base class for a special type of evolutionary operators that only
    works on populations.

    Args:
        in_size (int): The number of populations to input in the
            operator.
        out_size (int, optional): The number of populations returned by
            the operator. Defaults to ``in_size``.
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
    ):
        self._in_size = in_size
        self._out_size = out_size

    def process(
        self,
        populations: tuple[Population, ...],
    ) -> tuple[Population, ...]:
        """Processes the given individuals or populations with the
        operator. The number of objects to input and the size of the
        output is specified in the initialization of the operator.

        Args:
            objects (tuple[Population, ...]): A tuple of one or multiple
                populations to process.

        Returns:
            Processed population(s).
        """
        if not isinstance(populations, tuple):
            raise TypeError("Argument of operator has unknown type "
                            f"{type(populations)}")
        elif len(populations) != self._in_size:
            raise ValueError("Wrong number of arguments to operator "
                             f"({len(populations)}), "
                             f"should be {self._in_size}.")

        result = self._process(populations)
        if not isinstance(result, tuple):
            raise TypeError("Returned value of operator has unknown type "
                            f"{type(result)}")
        elif len(result) != self._out_size:
            raise ValueError("Wrong number of values returned by operator "
                             f"({len(result)}), should be {self._out_size}.")
        return result

    @abstractmethod
    def _process(
        self,
        populations: tuple[Population, ...],
    ) -> tuple[Population, ...]:
        ...
