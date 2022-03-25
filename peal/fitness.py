from typing import Any, Callable, Optional, Union
from peal.community import Community

from peal.genetics import GPTerminal
from peal.population import Individual, Population


class Fitness:
    """Class that is responsible for calculating the fitness of
    individuals in an environment.

    Args:
        method (callable): The method to be called for evaluating a
            single individual. This method should expect a value of
            type :class:`~peal.population.individual.Individual` and
            return a float value.
    """

    def __init__(self, method: Callable[[Individual], float]):
        self._method = method

    def evaluate(
        self,
        objects: Union[Individual, Population, Community],
    ) -> None:
        """Evaluates the fitness of individuals by changing their
        ``fitness`` attribute directly.

        Args:
            objects (Individual | Population | Community): A single
                individual, a population or a community to evaluate.
        """
        if isinstance(objects, Community):
            for pop in objects:
                for ind in pop:
                    ind.fitness = self._method(ind)
        if isinstance(objects, Population):
            for ind in objects:
                ind.fitness = self._method(ind)
        elif isinstance(objects, Individual):
            objects.fitness = self._method(objects)
        else:
            raise TypeError(f"Cannot evaluate object of type {type(objects)}")

    def __call__(self, population: Union[Individual, Population]) -> None:
        self.evaluate(population)


def fitness(method: Callable[[Individual], float]) -> Fitness:
    """Decorator for a fitness method.

    Declaring your own fitness function is possible with the class
    :class:`~peal.fitness.Fitness` or using this decorator
    on your evaluation method.
    The method you want to decorate will need to have the same arguments
    and return types as described in the mentioned class.
    """
    return Fitness(method=method)


def gp_evaluate(
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
                    raise RuntimeError(f"Argument name {name} "
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


class GPFitness(Fitness):
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
    ):
        eval_ = evaluation if evaluation is not None else (
            lambda array: float(array[0])
        )
        argsets = arguments if arguments is not None else [{}]
        super().__init__(lambda individual: eval_(
            [gp_evaluate(individual, argset) for argset in argsets]
        ))
