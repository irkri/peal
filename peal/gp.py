from typing import Any, Callable, Optional, get_type_hints

import numpy as np

from peal.fitness import Fitness
from peal.genetics import GenePool, GeneType
from peal.individual import Individual
from peal.operators.operator import Operator
from peal.population import Population


class GPNode:
    """Class representation of a node in a genetic programming tree."""

    __slots__ = ("rtype", "name")

    def __init__(self, rtype: type, name: str) -> None:
        self.rtype = rtype
        self.name = name


class GPCallable(GPNode):
    """Special GP tree node that represents a elementary function in
    such a tree with multiple arguments and a specific return type.
    """

    __slots__ = ("rtype", "name", "argtypes", "method")

    def __init__(
        self,
        rtype: type,
        name: str,
        argtypes: dict[str, Any],
        method: Callable[..., Any],
    ) -> None:
        super().__init__(rtype, name)
        self.argtypes = argtypes
        self.method = method

    def __call__(self, *args) -> Any:
        return self.method(*args)

    def __repr__(self) -> str:
        return f"{self.name}()"


class GPTerminal(GPNode):
    """Special GP tree node that represents a elementary function in
    such a tree with multiple arguments and a specific return type.
    """

    __slots__ = ("rtype", "name", "_value")

    def __init__(
        self,
        rtype: type,
        name: str,
        value: Optional[Any] = None,
    ) -> None:
        super().__init__(rtype, name)
        self._value = value

    @property
    def value(self) -> Any:
        if callable(self._value):
            self._value = self._value()
        return self._value

    @property
    def allocated(self) -> bool:
        return self.value is not None

    def copy(self) -> "GPTerminal":
        return GPTerminal(self.rtype, self.name, self._value)

    def __repr__(self) -> str:
        if not self.allocated:
            return f"<{self.name}>"
        return f"{self.value}"


class Pool(GenePool):
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
    ) -> None:
        super().__init__(typing=GeneType.NOMINAL)
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

    def _initialize(self, **kwargs) -> np.ndarray:
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
                # copying the terminal symbol is crucial for ephemeral
                # random constants; the true value of the terminal will
                # be set the first time it is accessed
                genes.append(terminal.copy())
            else:
                if rtype not in self._elementary:
                    raise IndexError("Failed to create a GP-based genom; "
                                     f"An elementary allele of type {rtype} "
                                     "is requested but not found.")
                elementary: GPCallable = np.random.choice(
                    np.array(self._elementary[rtype])
                )
                requested_types = elementary.argtypes
                genes.append(elementary)
                for vartype in requested_types.values():
                    stack.append((depth + 1, vartype))
        return np.array(genes)

    def allele(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator that can be used on a function to add a callable
        as an allele to the pool.
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
            self._terminal[type_].append(
                GPTerminal(rtype=type_, name=name),
            )

    def add_terminals(self, *terminals: Any) -> None:
        """Add terminal symbols to the list of alleles in this pool.
        If a callable function that takes no arguments is given, it is
        interpreted as a ephemeral random constant. If this terminal is
        drawn in the creation of a genetic programming tree, it will be
        called and the returned value will be placed in the tree.

        Args:
            terminals (Any): One or more values for terminal symbols.
        """
        for value in terminals:
            rtype = type(value()) if callable(value) else type(value)
            if rtype not in self._terminal:
                self._terminal[rtype] = []
            self._terminal[rtype].append(
                GPTerminal(rtype=rtype, name="", value=value),
            )


def evaluate(
    individual: Individual,
    arguments: Optional[dict[str, Any]] = None,
) -> Any:
    """Evaluates an individual that has an genetic programming tree-like
    genome. The evaluation executes all callables in their order they
    appear in the tree. If arguments (i.e. unallocated variables) are
    included in the individual, their values have to be specified in
    ``arguments``.

    Args:
        individual (Individual): The individual to evalutate.
        arguments (dict[str, Any], optional): A dictionary mapping
            argument names to values. Defaults to None.

    Returns:
        A value that represents the result of the tree evaluation.
    """
    argset = arguments if arguments is not None else {}
    if "x" not in argset:
        print(80*"=")
        print(f"{argset=}")
    values: list[Any] = []
    index = len(individual.genes) - 1
    while index >= 0:
        while isinstance(individual.genes[index], GPTerminal):
            if individual.genes[index].allocated:
                values.insert(0, individual.genes[index].value)
            else:
                name = individual.genes[index].name
                if name not in argset:
                    raise RuntimeError(f"Argument name {name!r} requested but "
                                       "not supplied")
                values.insert(0, argset[name])
            index -= 1
        argcount = len(individual.genes[index].argtypes)
        values.insert(
            0,
            individual.genes[index](*values[-argcount:])
        )
        values = values[:len(values)-argcount]
        index -= 1
    return values[0]


class Fitness(Fitness):
    """Fitness to use in a genetic programming process.
    The fitness will be the return value of the genome tree of
    operations an individual consists of.

    Args:
        arguments (list[dict[str, Any]], optional): A number of
            dictionaries mapping argument names to values of unallocated
            terminal symbols that genes of individuals might have.
            Defaults to empty list.
        evaluation (callable, optional): A function that returns a
            float by evaluating the return value of a individuals
            genetic programming tree. The return values are collected in
            lists and there are more than one value in this list if you
            supplied multiple dictionaries in ``arguments``, i.e. one
            value for each set of arguments given. Defaults to the float
            value of for an empty set of arguments.
    """

    def __init__(
        self,
        arguments: Optional[list[dict[str, Any]]] = None,
        evaluation: Optional[Callable[[list[Any]], float]] = None,
    ) -> None:
        eval_ = evaluation if evaluation is not None else (
            lambda array: float(array[0])
        )
        argsets = arguments if arguments is not None else [{}]
        super().__init__(lambda individual: eval_(
            [evaluate(individual, argset) for argset in argsets]
        ))


class PointMutation(Operator):
    """Point mutation used in a genetic programming algorithm.
    This mutation replaces a node in a genome tree by a subtree.

    Args:
        gene_pool (GPPool): The gene pool used to generate a genome
            tree for individuals.
        min_height (int, optional): The minimal height of the replacing
            subtree. Defaults to 1.
        max_height (int, optional): The maximal height of the replacing
            subtree. Defaults to 1.
        prob (float, optional): The probability to mutate one node in
            the tree representation of an individual. Defaults to 0.1.
    """

    def __init__(
        self,
        gene_pool: Pool,
        min_height: int = 1,
        max_height: int = 1,
        prob: float = 0.1,
    ) -> None:
        super().__init__()
        self._pool = gene_pool
        self._min_height = min_height
        self._max_height = max_height
        self._prob = prob

    def _process_population(
        self,
        container: Population,
    ) -> Population:
        if np.random.random_sample() >= self._prob:
            return container.deepcopy()

        ind = container[0].copy()
        index = np.random.randint(0, len(ind.genes))
        # search for subtree slice starting at index in the tree
        right = index + 1
        total = 0
        if not isinstance(ind.genes[index], GPTerminal):
            total = len(ind.genes[index].argtypes)
        while total > 0:
            if isinstance(ind.genes[right], GPTerminal):
                total -= 1
            else:
                total -= len(ind.genes[right].argtypes) - 1
            right += 1
        ind.genes = np.concatenate((
            ind.genes[:index],
            self._pool.create_genome(
                rtype=ind.genes[index].rtype,
                height=np.random.randint(self._min_height, self._max_height+1),
            ),
            ind.genes[right:],
        ))
        return Population(ind)
