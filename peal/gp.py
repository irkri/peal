from abc import abstractmethod
from typing import Any, Callable, Optional, Union, get_type_hints
from winreg import REG_OPTION_NON_VOLATILE

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

    @abstractmethod
    def value(self, **kwargs) -> Any:
        ...

    @abstractmethod
    def copy(self) -> "GPNode":
        ...


class GPCallable(GPNode):
    """Special GP tree node that represents a elementary function in
    such a tree with multiple arguments and a specific return type.
    """

    __slots__ = ("argtypes", "method", "children")

    def __init__(
        self,
        rtype: type,
        name: str,
        argtypes: list[type],
        method: Callable[..., Any],
        children: Optional[list[GPNode]] = None
    ) -> None:
        super().__init__(rtype, name)
        self.argtypes = argtypes
        self.method = method
        self.children: list[GPNode] = [] if children is None else children

    def value(self, **kwargs) -> Any:
        return self.method(*(node.value(**kwargs) for node in self.children))

    def copy(self) -> "GPCallable":
        return GPCallable(
            self.rtype,
            self.name,
            self.argtypes,
            self.method,
            [child.copy() for child in self.children]
        )

    def __repr__(self) -> str:
        return f"{self.name}({', '.join(map(str, self.children))})"


class GPTerminal(GPNode):
    """Special GP tree node that represents a elementary function in
    such a tree with multiple arguments and a specific return type.
    """

    __slots__ = ("_value", )

    def __init__(
        self,
        rtype: type,
        name: str,
        value: Optional[Any] = None,
    ) -> None:
        super().__init__(rtype, name)
        self._value = value

    def value(self, **kwargs) -> Any:
        if callable(self._value):
            self._value = self._value()
        if not self.allocated:
            if not self.name in kwargs:
                raise RuntimeError(
                    f"Found unbound terminal named {self.name!r} "
                    "without a value given"
                )
            return kwargs[self.name]
        return self._value

    @property
    def allocated(self) -> bool:
        return self._value is not None

    def copy(self) -> "GPTerminal":
        return GPTerminal(self.rtype, self.name, self._value)

    def __repr__(self) -> str:
        if not self.allocated:
            return f"<{self.name}>"
        return f"{self.value()}"


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

    def _fill_argtypes(
        self,
        node: GPNode,
        terminal_prob: float,
    ) -> None:
        if not isinstance(node, GPCallable):
            return
        for rtype in node.argtypes:
            if rtype not in self._elementary and rtype not in self._terminal:
                raise RuntimeError(
                    "Failed to create a GP-based genome; An allele of "
                    f"type {rtype} is requested but not found."
                )
            if (np.random.random() < terminal_prob
                    or rtype not in self._elementary):
                if rtype not in self._terminal:
                    raise IndexError(
                        "Failed to create a GP-based genome; "
                        f"A terminal allele of type {rtype} "
                        "is requested but not found."
                    )
                node.children.append(np.random.choice(
                    np.array(self._terminal[rtype])
                ).copy())
            else:
                if rtype not in self._elementary:
                    raise IndexError(
                        "Failed to create a GP-based genom; "
                        f"An elementary allele of type {rtype} "
                        "is requested but not found."
                    )
                node.children.append(np.random.choice(
                    np.array(self._elementary[rtype])
                ).copy())

    def _initialize(self, **kwargs) -> np.ndarray:
        rtype = kwargs.get(
            "rtype",
            np.random.choice(np.array(list(self._elementary.keys()))),
        )
        height = kwargs.get(
            "height",
            np.random.randint(self._min_depth, self._max_depth),
        )
        if height == 0:
            return np.random.choice(np.array(self._terminal[rtype])).copy()
        root: GPCallable = np.random.choice(
            np.array(self._elementary[rtype])
        ).copy()
        node = root
        terminal_prob = (
            len(self._terminal)
            / (len(self._terminal) + len(self._elementary))
        )
        nodes = [node]
        for _ in range(height-1):
            next_nodes = []
            for n in nodes:
                self._fill_argtypes(n, terminal_prob=terminal_prob)
                if isinstance(n, GPCallable):
                    for child in n.children:
                        next_nodes.append(child)
            nodes = next_nodes
        for n in nodes:
            self._fill_argtypes(n, terminal_prob=1.0)
        return np.array([root])

    def allele(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator that can be used on a function to add a callable
        as an allele to the pool.
        """
        hints = get_type_hints(func)
        rtype = hints.pop("return")

        if rtype not in self._elementary:
            self._elementary[rtype] = []
        self._elementary[rtype].append(GPCallable(
            rtype=rtype,
            name=func.__name__,
            argtypes=list(hints.values()),
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
    return individual.genes[0].value(**argset)


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


def _get_path_to_random_child(node: GPCallable) -> list[int]:
    # goes a random path from node to a leaf and returns the indices
    # selected on that path
    path = []
    while isinstance(node, GPCallable):
        child_index = np.random.randint(len(node.children))
        path.append(child_index)
        node = node.children[child_index]
    index = np.random.randint(-1, len(path))
    return path[:(index+1)]


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

    def _process_population(self, container: Population) -> Population:
        if np.random.random_sample() >= self._prob:
            return Population(
                Individual(np.array([container[0].genes[0].copy()]))
            )

        root: GPNode = container[0].genes[0].copy()
        path_to_child = _get_path_to_random_child(root)
        if not isinstance(root, GPCallable) or not path_to_child:
            return Population(Individual(self._pool.create_genome(
                rtype=root.rtype,
                height=np.random.randint(self._min_height, self._max_height+1),
            )))
        node = root
        for index in path_to_child[:-1]:
            node = node.children[index]
        node.children[path_to_child[-1]] = self._pool.create_genome(
            rtype=node.children[path_to_child[-1]].rtype,
            height=np.random.randint(self._min_height, self._max_height+1),
        )[0]
        return Population(Individual(np.array([root])))


class Crossover(Operator):

    def __init__(self, probability: float) -> None:
        self._probability = probability

    def _process_population(self, container: Population) -> Population:
        return super()._process_population(container)
