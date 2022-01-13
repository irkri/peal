from abc import ABC, abstractmethod
from typing import Optional, Union

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
        iter_type (IterationType, optional): The type of iteration
            mechanism to use to iterate over a given population.
            Defaults to
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

    def process(
        self,
        objects: Union[tuple[Individual, ...], Population],
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
        if isinstance(objects, Population):
            return self.process_all(objects)

        if not isinstance(objects, tuple):
            raise TypeError("Argument of operator has unknown type "
                            f"{type(objects)}")
        elif len(objects) != self._in_size:
            raise ValueError("Wrong number of arguments to operator "
                             f"({len(objects)}), should be {self._in_size}.")

        result = self._process(objects)
        if not isinstance(result, tuple):
            raise TypeError("Return of operator has unknown type "
                            f"{type(objects)}")
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
        objects: tuple[Individual, ...],
    ) -> tuple[Individual, ...]:
        ...

    def __call__(
        self,
        objects: Union[tuple[Individual, ...], Population],
    ) -> Union[tuple[Individual, ...], Population]:
        return self.process(objects)


class PopulationOperator(ABC):
    """Base class for a special type evolutionary operators that only
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
        objects: tuple[Population, ...],
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
        if not isinstance(objects, tuple):
            raise TypeError("Argument of operator has unknown type "
                            f"{type(objects)}")
        elif len(objects) != self._in_size:
            raise ValueError("Wrong number of arguments to operator "
                             f"({len(objects)}), should be {self._in_size}.")

        result = self._process(objects)
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
        objects: tuple[Population, ...],
    ) -> tuple[Population, ...]:
        ...

    def __call__(
        self,
        objects: tuple[Population, ...],
    ) -> tuple[Population, ...]:
        return self.process(objects)
