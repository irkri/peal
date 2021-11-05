"""Standard configuration module for all peal operations."""

import inspect
from functools import wraps
from typing import Any, Callable, Union

from peal.environment.population import Population


OPERATION_MARK = "peal_op"

_Toperation = Callable[..., Population]
_Toperationundetermined = Callable[..., Union[Population, _Toperation]]
_Toperationdecorator = Callable[
    [Callable[..., Population]],
    _Toperationundetermined
]

reproductions: dict[str, _Toperationundetermined] = dict()
mutations: dict[str, _Toperationundetermined] = dict()
selections: dict[str, _Toperationundetermined] = dict()

CATEGORIES = {
    "reproduction": reproductions,
    "mutation": mutations,
    "selection": selections,
}


def check_marked(operator: _Toperationundetermined) -> str:
    if OPERATION_MARK not in operator.__dict__:
        raise ValueError("Invalid operator. Did you use a decorator?")
    return operator.__dict__[OPERATION_MARK]


def _check_operator_arguments(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    default_kwargs: dict[str, tuple[type, Any]],
) -> Union[dict[str, Any], Population]:
    """Checks the arguments passed to an operator.
    This allows for an easy definition of operators that can be used
    with or without optional arguments. If the first argument is a
    population, this function returns this population. Otherwise it
    returns the given arguments if they match the types of provided
    arguments in ``defaults_kwargs``.

    Args:
        args (tuple[Any]): A list of positional arguments that were
            passed to the operator.
        kwargs (dict[str, Any]): Named arguments that were passed to
            the operator.
        default_kwargs (dict[str, tuple[type, Any]]): A dictionary
            mapping argument names to their expected type and default
            value.

    Returns:
        (kwargs | Population): All allowed arguments that were passed to
            the operator or the Population on which it was called on.
    """
    if len(args) > 0 and isinstance(args[0], Population):
        # operator was called on the population
        return args[0]

    kwargs_ = kwargs.copy()

    for name in kwargs_.keys():
        if name not in default_kwargs.keys():
            raise ValueError(f"Unknown argument {name!r} given to the "
                             "evolutionary operator")

    argnames = iter(default_kwargs)
    for arg in args:
        argname = next(argnames)
        if isinstance(arg, default_kwargs[argname][0]):
            kwargs_[argname] = arg
        else:
            raise RuntimeError("Cannot determine the given positional "
                               "argument. Use named arguments instead.")

    return kwargs_


def _register(category: str, function: _Toperation):
    if function.__name__ in CATEGORIES[category]:
        raise RuntimeError(f"{category} operator with same name already "
                           "registered")
    CATEGORIES[category][function.__name__] = function
    function.__dict__[OPERATION_MARK] = category


def operation(category: str) -> _Toperationdecorator:
    """This decorator allows for declaring an evolutionary operation.
    It has to be called with the specific category argument.

    Args:
        category (str): The category of which the operation belongs to.
            Has to be either "mutation", "reproduction" or "selection".

    Returns:
        _Toperation: An operator.
    """
    if category not in CATEGORIES.keys():
        raise ValueError(f"Unknown category: {category}")

    def _selection(func) -> _Toperationundetermined:
        defaults = {
            name: value.default
            for name, value in inspect.signature(func).parameters.items()
            if value.default is not inspect.Parameter.empty
        }
        _register(category, func)

        @wraps(func)
        def wrapper(*args, **kwargs) -> Union[Population, _Toperation]:
            given = _check_operator_arguments(args, kwargs, defaults)
            if isinstance(given, Population):
                return func(given)

            @wraps(func)
            def _blueprint(population: Population) -> Population:
                return func(population, **given)
            return _blueprint
        return wrapper

    return _selection
