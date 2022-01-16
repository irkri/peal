from abc import ABC, abstractmethod
from typing import Generic, TypeVar, get_args

from peal.community import Community
from peal.operations.iteration import IterationType
from peal.population import Population

T_operation = TypeVar("T_operation", Population, Community)


class Operator(ABC, Generic[T_operation]):
    """Base class for evolutionary operators in peal.

    Args:
        in_size (int): The number of individuals to input in the
            operator.
        out_size (int): The number of individuals returned by
            the operator.
        iter_type (IterationType, optional): The iteration type of the
            iterator. Defaults to
            :class:`~peal.operations.iteration.SingleIteration`.
    """

    def __init__(
        self,
        iter_type: IterationType[T_operation],
    ):
        self._iter_type: IterationType[T_operation] = iter_type

    @property
    def iter_type(self) -> IterationType[T_operation]:
        """The type of iteration through a population to use if the
        operators :meth:`process` method is called on a population. This
        leads to the processing of single individuals in the population.
        In which order they are processed or how many individuals are
        processed at once will be specified by such an iteration type.
        """
        return self._iter_type

    @iter_type.setter
    def iter_type(self, iter_type: IterationType[T_operation]) -> None:
        if not isinstance(iter_type, IterationType):
            raise TypeError(f"Expected IterationType, got {type(iter_type)}")
        self._iter_type = iter_type

    def process(
        self,
        container: T_operation,
    ) -> T_operation:
        """Processes the given population or community with the
        operator.

        Args:
            container (Population | Community): A tuple of
                one or multiple populations to process. Only one
                operation is performed.

        Returns:
            Processed population or community.
        """
        iteration = self._iter_type.iterate(container)
        new_container: T_operation = (
            get_args(self.__orig_bases__[0])[0]()  # type: ignore
        )
        for batch in iteration:
            new_container.integrate(self._process(batch))
        return new_container

    @abstractmethod
    def _process(
        self,
        container: T_operation,
    ) -> T_operation:
        ...
