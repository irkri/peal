from abc import abstractmethod
from typing import Any, Callable, Optional, get_type_hints

import numpy as np

from peal.genetics import GenePool, GeneType
from peal.individual import Individual
from peal.operators.iteration import StraightIteration
from peal.operators.operator import Operator
from peal.population import Population


class GPNode:
    """Class representation of a node in a genetic programming tree."""

    __slots__ = ("rtype", "name", "children")

    def __init__(self, rtype: type, name: str) -> None:
        self.rtype = rtype
        self.name = name
        self.children: list[GPNode] = []

    @abstractmethod
    def value(self, **kwargs) -> Any:
        ...

    @abstractmethod
    def copy(self) -> "GPNode":
        ...

    @abstractmethod
    def height(self) -> int:
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

    def height(self) -> int:
        return max(child.height() for child in self.children) + 1

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

    def height(self) -> int:
        return 0

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
        max_height (int): The maximum height of a genome tree.
    """

    def __init__(self, max_height: int) -> None:
        super().__init__(typing=GeneType.NOMINAL)
        self._elementary: dict[type, list[GPCallable]] = {}
        self._terminal: dict[type, list[GPTerminal]] = {}
        self._max_height = max_height

    @property
    def max_height(self) -> int:
        return self._max_height

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
            np.random.randint(self.max_height),
        )
        if height == 0:
            return np.array(
                [np.random.choice(np.array(self._terminal[rtype])).copy()]
            )
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


def _get_path_to_random_child(
    node: GPNode,
    max_depth: Optional[int] = None,
) ->  list[int]:
    # goes a random path from node to a leaf and returns the indices
    # selected on that path as well as the node arrived at
    if max_depth is None:
        max_depth = 100_000
    path = []
    while isinstance(node, GPCallable) and len(path) < max_depth:
        child_index = np.random.randint(len(node.children))
        path.append(child_index)
        node = node.children[child_index]
    index = np.random.randint(-1, len(path))
    return path[:(index+1)]


class PointMutation(Operator):
    """Point mutation used in a genetic programming algorithm.
    This mutation replaces a node in a genome tree by a subtree.

    Args:
        prob (float, optional): The probability to mutate one node in
            the individuals genome tree. Defaults to ``0.1``.
    """

    def __init__(self, prob: float = 0.1) -> None:
        super().__init__()
        self._prob = prob

    def _process_population(
        self,
        container: Population,
        /, *,
        pool: Optional[GenePool] = None,
    ) -> Population:
        if not isinstance(pool, Pool):
            raise TypeError("Unknown gene pool given to the operator")
        if np.random.random_sample() >= self._prob:
            return Population(
                Individual(np.array([container[0].genes[0].copy()]))
            )

        root: GPCallable = container[0].genes[0].copy()
        path_to_child = _get_path_to_random_child(root)
        if not isinstance(root, GPCallable) or not path_to_child:
            return Population(Individual(pool.create_genome(
                rtype=root.rtype,
                height=np.random.randint(pool.max_height+1),
            )))
        node = root
        for index in path_to_child[:-1]:
            node = node.children[index]
        node.children[path_to_child[-1]] = pool.create_genome(
            rtype=node.children[path_to_child[-1]].rtype,
            height=np.random.randint(
                pool.max_height-len(path_to_child)+1,
            )
        )[0]
        return Population(Individual(np.array([root])))


class Crossover(Operator):

    def __init__(self, probability: float = 0.7) -> None:
        super().__init__(StraightIteration(batch_size=2))
        self._probability = probability

    def _process_population(
        self,
        container: Population,
        /, *,
        pool: Optional[GenePool] = None,
    ) -> Population:
        if not isinstance(pool, Pool):
            raise TypeError("Unknown gene pool given to the operator")
        if np.random.random() <= self._probability:
            return container.copy()

        ind1, ind2 = container
        path1 = _get_path_to_random_child(ind1.genes[0])
        node1 = ind1.genes[0]
        for index in path1[:-1]:
            node1 = node1.children[index]
        path2 = _get_path_to_random_child(
            ind2.genes[0],
            max_depth=pool.max_height-node1.height()-1,
        )
        node2 = ind2.genes[0]
        for index in path2[:-1]:
            node2 = node2.children[index]
        temp1 = node1 if not path1 else node1.children[path1[-1]]
        temp2 = node2 if not path2 else node2.children[path2[-1]]
        if temp1.rtype != temp2.rtype:
            return container.copy()

        if path1 and path2:
            if temp2.height() + len(path1) <= pool.max_height:
                node1.children[path1[-1]] = node2.children[path2[-1]].copy()
            node2.children[path2[-1]] = temp1.copy()
        elif path1 and not path2:
            if temp2.height() + len(path1) <= pool.max_height:
                node1.children[path1[-1]] = node2.copy()
            ind2 = Individual(np.array([temp1.copy()]))
        elif not path1 and path2:
            ind1 = Individual(np.array([node2.children[path2[-1]].copy()]))
            node2.children[path2[-1]] = temp1.copy()
        return Population((ind1, ind2))
