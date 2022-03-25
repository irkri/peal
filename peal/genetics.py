from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Optional, Union, get_type_hints

import numpy as np


class GeneType(Enum):
    """Enum class for a category of a gene type.
    Availalbe types are:

        * ``NOMINAL``
        * ``ORDINAL``
        * ``METRIC``
        * ``CONST_SIZE``

    This way, non-comparable types in Python are forbidden.
    """

    NOMINAL = auto()
    ORDINAL = auto()
    METRIC = auto()

    CONST_SIZE = auto()


class GenePool(ABC):
    """Base class for any gene pool in PEAL. A gene pool is a collection
    of alleles for genes an individual can have. With this information,
    a gene pool knows all possible genomes individuals in a population
    for a given problem can have.

    Args:
        typing (tuple[GeneType, ...]): The typing signature for this
            gene pool. Have a look at the enum class :class:`GeneType`
            for details.
    """

    def __init__(self, typing: tuple[GeneType, ...]):
        self._size = 0 if GeneType.METRIC in typing else np.inf
        self._typing = typing

    @property
    def typing(self) -> tuple[GeneType, ...]:
        """The typing information of this gene pool."""
        return self._typing

    @property
    def size(self) -> Union[int, float]:
        """The number of different alleles in this gene pool."""
        return self._size

    @abstractmethod
    def random_genome(self, **kwargs) -> np.ndarray:
        """Generates a numpy array containing randomly selected alleles
        that are pulled from this gene pool. This method helps creating
        new individuals in a population using a
        :class:`~peal.population.breeding.Breeder`.

        Returns:
            np.ndarray: An array representing the genome of an
                individual.
        """


class IntegerPool(GenePool):
    """A gene pool of constant length genomes only containing integers
    in the given range.

    Args:
        shape (int): The number of genes in a genome.
        lower (int): The smallest integer one gene can be.
        upper (int): The largest integer one gene can be.
    """

    def __init__(
        self,
        shape: int,
        lower: int,
        upper: int,
    ):
        super().__init__(typing=(GeneType.ORDINAL, GeneType.CONST_SIZE))
        self.lower = lower
        self.upper = upper
        self._shape = shape
        self._size = upper - lower + 1

    def random_genome(self, **kwargs) -> np.ndarray:
        return np.random.randint(
            self.lower,
            self.upper + 1,
            size=self._shape
        )


class NumberPool(GenePool):
    """A gene pool that supports all floats and integers in a specified
    range.

    Args:
        shape (int): The number of genes in a genome.
        lower (int | float): The lower bound for all genes.
        upper (int | float): The upper bound for all genes.
    """

    def __init__(
        self,
        shape: int,
        lower: Union[int, float],
        upper: Union[int, float],
    ):
        super().__init__(typing=(GeneType.METRIC, GeneType.CONST_SIZE))
        self.lower = lower
        self.upper = upper
        self._shape = shape

    def random_genome(self, **kwargs) -> np.ndarray:
        return (
            (self.upper - self.lower)
            * np.random.random_sample(size=self._shape)
            + self.lower
        )


@dataclass
class GPNode:
    """Class representation of a node in a genetic programming tree."""

    rtype: type
    name: str


@dataclass
class GPCallable(GPNode):
    """Special GP tree node that represents a elementary function in
    such a tree with multiple arguments and a specific return type.
    """

    argtypes: dict[str, Any]
    method: Callable[..., Any]

    def __call__(self, *args) -> Any:
        return self.method(*args)


@dataclass
class GPTerminal(GPNode):
    """Special GP tree node that represents a terminal symbol, e.g. as
    an argument for :class:`GPCallable` nodes.
    """

    value: Optional[Any] = None

    @property
    def allocated(self) -> bool:
        return self.value is not None


class GPPool(GenePool):
    """A gene pool that is used for genetic programming. An individual
    in GP has a tree-like structure of genes and single nodes in this
    tree are methods listed in this type of pool.
    New methods should be added by using the decorator
    :meth:`GPPool.allele` on a newly created instance of this class.

    Args:
        min_depth (int): The minimum depth of a genome tree.
        max_depth (int): The maximum depth of a genome tree.
    """

    def __init__(
        self,
        min_depth: int,
        max_depth: int,
    ):
        super().__init__(typing=(GeneType.NOMINAL, ))
        self._elementary: dict[type, list[GPCallable]] = {}
        self._terminal: dict[type, list[GPTerminal]] = {}
        self._min_depth = min_depth
        self._max_depth = max_depth

    @property
    def min_depth(self) -> int:
        return self._min_depth

    @property
    def max_depth(self) -> int:
        return self._max_depth

    def random_genome(self, **kwargs) -> np.ndarray:
        rtype = kwargs.get(
            "rtype",
            np.random.choice(np.array(list(self._elementary.keys()))),
        )
        height = kwargs.get(
            "height",
            np.random.randint(self._min_depth, self._max_depth),
        )
        stack: list[tuple[int, type]] = [(0, rtype)]
        genes = []
        while len(stack) > 0:
            depth, rtype = stack.pop(0)
            if depth == height:
                if rtype not in self._terminal:
                    raise IndexError("Failed to create a GP-based genome; "
                                     f"A terminal allele of type {rtype} "
                                     "is requested but not found.")
                terminal = np.random.choice(np.array(self._terminal[rtype]))
                genes.append(terminal)
            else:
                if rtype not in self._elementary:
                    raise IndexError("Failed to create a GP-based genom; "
                                     f"An elementary allele of type {rtype} "
                                     "is requested but not found.")
                elementary = np.random.choice(
                    np.array(self._elementary[rtype])
                )
                requested_types = elementary.__dict__["argtypes"]
                genes.append(elementary)
                for vartype in requested_types.values():
                    stack.append((depth + 1, vartype))
        return np.array(genes)

    def allele(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator that can be used on a function to add a callable
        as an allele to the pool.
        Callables with no arguments are counted as terminal symbols of
        the individuals representation.
        """
        hints = get_type_hints(func).copy()
        rtype = hints.pop("return")

        if rtype not in self._elementary:
            self._elementary[rtype] = []
        self._elementary[rtype].append(GPCallable(
            rtype=rtype,
            name=func.__name__,
            argtypes=hints,
            method=func,
        ))
        return func

    def add_arguments(self, arguments: dict[str, type]) -> None:
        """Add special terminal symbols, i.e. arguments, to the list of
        alleles in this pool.
        These terminal symbols do not have a fixed value but can be seen
        as arguments that have to be supplied to each individual before
        calculating its fitness.

        Args:
            arguments (dict[str, type]): A dictionary mapping argument
                names to their corresponding type.
        """
        for name, type_ in arguments.items():
            if type_ not in self._terminal:
                self._terminal[type_] = []
            self._terminal[type_].append(GPTerminal(
                rtype=type_,
                name=name,
            ))

    def add_terminals(self, terminals: list[Any]) -> None:
        """Add terminal symbols to the list of alleles in this pool.

        Args:
            arguments (list[Any]): A list of terminal symbols.
        """
        for value in terminals:
            var = GPTerminal(
                rtype=type(value),
                name="",
                value=value,
            )
            if type(value) not in self._terminal:
                self._terminal[type(value)] = []
            self._terminal[type(value)].append(var)
