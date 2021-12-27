from abc import ABC, abstractmethod
from functools import wraps
from typing import Callable, Union, get_type_hints

import numpy as np


class GenePool(ABC):
    """Base class for any gene pool in PEAL. A gene pool is a collection
    of alleles for genes an individual can have. With this information,
    a gene pool knows all possible genomes individuals in a population
    for a given problem can have.

    Args:
        shape (tuple[int] | int): The shape of a genome for a single
            individual. Typically, this is an integer denoting the
            number of genes the individual has.
    """

    def __init__(self, shape: Union[tuple[int], int]):
        if isinstance(shape, tuple):
            raise NotImplementedError
        self._shape = shape

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
    """A gene pool only containing integers in the given range.

    Args:
        shape (tuple[int] | int): The shape of a genome for a single
            individual. Typically, this is an integer denoting the
            number of genes the individual has.
        lower (int): The smallest integer that an allele in this pool
            can be.
        upper (int): The largest integer that an allele in this pool
            can be.
    """

    def __init__(self, shape: Union[tuple[int], int], lower: int, upper: int):
        super().__init__(shape)
        self._lower = lower
        self._upper = upper

    def random_genome(self, **kwargs) -> np.ndarray:
        return np.random.randint(self._lower, self._upper, size=self._shape)


class NumberPool(GenePool):
    """A gene pool that supports all floats and integers in a specified
    range.

    Args:
        shape (tuple[int] | int): The shape of a genome for a single
            individual. Typically, this is an integer denoting the
            number of genes the individual has.
        lower (int | float): The lower bound of the range of numbers in
            this pool.
        upper (int | float): The upper bound of the range of numbers in
            this pool.
    """

    def __init__(
        self,
        shape: Union[tuple[int], int],
        lower: Union[int, float],
        upper: Union[int, float],
    ):
        super().__init__(shape)
        self._lower = lower
        self._upper = upper

    def random_genome(self, **kwargs) -> np.ndarray:
        return (
            (self._upper-self._lower)
            * np.random.random_sample(size=self._shape)
            + self._lower
        )


class GPPool(GenePool):
    """A gene pool that is used for genetic programming. An individual
    in GP has a tree-like structure of genes and single nodes in this
    tree are methods listed in this type of pool.
    New methods should be added by using the decorator
    :meth:`GPPool.allele` on a newly created instance of this class.

    Args:
        shape (tuple[int] | int): The shape of a genome for a single
            individual. In the case of a genome for genetic programming,
            this argument will be intepreted as the minimum depth of a
            genome tree and individual created with this pool can have.
        max_depth (int): The maximum depth of any genome tree in this
            gene pool.
    """

    def __init__(
        self,
        shape: Union[tuple[int], int],
        max_depth: int,
    ):
        super().__init__(shape)
        self._elementary: dict[type, list[Callable]] = dict()
        self._terminal: dict[type, list[Callable]] = dict()
        self._max_depth = max_depth

    def random_genome(self, **kwargs) -> np.ndarray:
        rtype = kwargs.get(
            "rtype",
            np.random.choice(list(self._elementary.keys())),
        )
        height = np.random.randint(self._shape, self._max_depth)
        stack = [(0, rtype)]
        genes = []
        while len(stack) > 0:
            depth, rtype = stack.pop(0)
            if depth == height:
                if rtype not in self._terminal:
                    raise IndexError("Failed to create a GP-based genom; "
                                     f"A terminal allele of type {rtype} "
                                     "is requested but not found.")
                terminal = np.random.choice(self._terminal[rtype])
                genes.append(terminal)
            else:
                if rtype not in self._elementary:
                    raise IndexError("Failed to create a GP-based genom; "
                                     f"An elementary allele of type {rtype} "
                                     "is requested but not found.")
                elementary = np.random.choice(self._elementary[rtype])
                requested_types = elementary.__dict__["argtypes"]
                genes.append(elementary)
                for vartype in requested_types.values():
                    stack.append((depth + 1, vartype))
        return np.array(genes)

    def allele(self, func: Callable) -> Callable:
        """Decorator that can be used on a function to add a callable
        as an allele to the pool.
        Callables with no arguments are counted as terminal symbols of
        the individuals representation.
        """
        hints = get_type_hints(func).copy()
        rtype = hints.pop("return")
        func.__dict__["rtype"] = rtype
        func.__dict__["argtypes"] = hints
        if func.__code__.co_argcount == 0:
            func.__dict__["terminal"] = True
            if rtype not in self._terminal:
                self._terminal[rtype] = [func]
            else:
                self._terminal[rtype].append(func)
        else:
            func.__dict__["terminal"] = False
            if rtype not in self._elementary:
                self._elementary[rtype] = [func]
            else:
                self._elementary[rtype].append(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            if len(kwargs) > 0:
                raise ValueError("Keyword arguments are not supported for "
                                 "elementary alleles of genetic programming "
                                 "individuals.")
            func(*args)
        return wrapper

    def configure(self, min_depth: int, max_depth: int) -> "GPPool":
        """Reconfigures the GPPool object. This method can be used for
        convenience to reuse the same GPPool instance as an argument to
        a :class:`~peal.population.breeding.Breeder` as well as
        supplying it to an evolutionary operator, e.g.
        :class:`~peal.operations.mutation.gp.GPPoint`, but each time
        with different arguments ``min_depth`` and ``max_depth``.

        Args:
            min_depth (int): The minimum depth of a genome tree an
                individual created with this pool can have.
            max_depth (int): The maximum depth of any genome tree in this
                gene pool.

        Returns:
            GPPool: This instance of the GPPool.
        """
        self._shape = min_depth
        self._max_depth = max_depth
        return self
