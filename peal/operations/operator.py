from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from peal.community import Community
from peal.operations.iteration import IterationType
from peal.population import Population

T_operation = TypeVar("T_operation", Population, Community)


class Operator(ABC, Generic[T_operation]):
    """Base class for evolutionary operators in peal. An operator is
    normally called on a :class:`~peal.population.Population` or a
    :class:`~peal.community.Community`. Smaller parts of such a
    container are selected and feed forward to the protected method
    :meth:`Operator._process` that has to be overwritten by inheriting
    classes. The property ``iter_type`` of an operator is used to
    iterate over the input population or community, return correspoding
    smaller populations or communities, which then are given to the
    metioned protected method. All results will be concatenated and
    returned by the public :meth:`Operator.process` method.

    New operators should always inherit from either
    :class:`PopulationOperator` or :class:`CommunityOperator`.

    Args:
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

    @abstractmethod
    def process(
        self,
        container: T_operation,
    ) -> T_operation:
        """Processes and returns the given population or community."""

    @abstractmethod
    def _process(
        self,
        container: T_operation,
    ) -> T_operation:
        ...


class PopulationOperator(Operator[Population]):
    """Base class for evolutionary operators that operate on groups of
    individuals in a population.

    For more information, check the base class :class:`Operator`.
    """

    def process(
        self,
        container: Population,
    ) -> Population:
        iteration = self._iter_type.iterate(container)
        new_container = Population()
        for batch in iteration:
            new_container.integrate(self._process(batch))
        return new_container


class CommunityOperator(Operator[Community]):
    """Base class for evolutionary operators that operate on groups of
    populations in a community.

    For more information, check the base class :class:`Operator`.
    """

    def process(
        self,
        container: Community,
    ) -> Community:
        iteration = self._iter_type.iterate(container)
        new_container = Community()
        for batch in iteration:
            new_container.integrate(self._process(batch))
        return new_container
