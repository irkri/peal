from typing import Any, Callable, Optional, Union

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

    def evaluate(self, population: Union[Individual, Population]):
        """Evaluates the fitness of all individuals in the given
        population. This method changes attributes of individuals in
        ``population`` directly.

        Args:
            population (Population | Individual): The Population or a
                single individual to evaluate.
        """
        if isinstance(population, Population):
            for ind in population:
                if not ind.fitted:
                    ind.fitness = self._method(ind)
        elif isinstance(population, Individual):
            population.fitness = self._method(population)

    def __call__(self, population: Union[Individual, Population]):
        self.evaluate(population)


def fitness(method: Callable[[Individual], float]) -> Fitness:
    """Decorator for a fitness method.

    Declaring your own fitness function is possible with the class
    :class:`~peal.evaluation.fitness.Fitness` or using this decorator
    on your evaluation method.
    The method you want to decorate will need to have the same arguments
    and return types as described in the mentioned class.
    """
    return Fitness(method=method)


def gp_evaluate(
    individual: Individual,
    arguments: list[dict[str, Any]],
) -> list[float]:
    values: list[list[Any]] = [[] for _ in range(len(arguments))]
    for i, argset in enumerate(arguments):
        index = len(individual.genes) - 1
        while index >= 0:
            while isinstance(individual.genes[index], GPTerminal):
                if individual.genes[index].allocated:
                    values[i].insert(0, individual.genes[index].value)
                else:
                    name = individual.genes[index].name
                    if name not in argset:
                        raise RuntimeError(f"Argument name {name} "
                                            "not supplied")
                    values[i].insert(0, argset[name])
                index -= 1
            argcount = len(individual.genes[index].argtypes)
            values[i].insert(
                0,
                individual.genes[index](*values[i][-argcount:])
            )
            values[i] = values[i][:len(values[i])-argcount]
            index -= 1
    return [v[0] for v in values]


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
        arguments = arguments if arguments is not None else [dict()]
        super().__init__(lambda individual: eval_(
            gp_evaluate(individual, arguments)
        ))
