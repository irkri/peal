"""Standard configuration module for all peal operations."""

import inspect
from functools import wraps
from typing import Any, Callable, Union

from peal.population import Population


OPERATION_MARK = "peal_op"
FIXED_OPERATION_MARK = "peal_fixed"

_Toperation = Callable[[Population], Population]
_Toperationundetermined = Callable[..., _Toperation]
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


def check_operation(operator: Callable) -> str:
    """Checks if the given callable was successfully decorated and fixed
    as an operation.
    This can be done be using the decorator :meth:`operation`.

    Args:
        operator (callable): The function to check.

    Raises:
        ValueError: If at least one of the mentioned criteria are not
            fulfilled.

    Returns:
        str: Category the decorator belongs to.
    """
    if (OPERATION_MARK not in operator.__dict__
            or FIXED_OPERATION_MARK not in operator.__dict__
            or operator.__name__ not in CATEGORIES[
                operator.__dict__[OPERATION_MARK]]
            or not operator.__dict__[FIXED_OPERATION_MARK]):
        raise ValueError("Invalid or non-fixed operator. Did you use a "
                         "decorator for declaring the operation?")
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


def _register(category: str, function: _Toperationundetermined):
    if function.__name__ in CATEGORIES[category]:
        raise RuntimeError(f"{category} operator with same name already "
                           "registered")
    CATEGORIES[category][function.__name__] = function
    function.__dict__[OPERATION_MARK] = category
    function.__dict__[FIXED_OPERATION_MARK] = False


def _register_as_fixed(function: _Toperation):
    if OPERATION_MARK not in function.__dict__:
        raise TypeError("Function is not a registered operation")
    function.__dict__[FIXED_OPERATION_MARK] = True


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
        def wrapper(*args, **kwargs) -> _Toperation:
            given = _check_operator_arguments(args, kwargs, defaults)
            if isinstance(given, Population):
                raise RuntimeError("Cannot call undetermined operation. "
                                   "Please call the operator once before "
                                   "passing it to a process.")
            _register_as_fixed(func)

            @wraps(func)
            def _operation_blueprint(population: Population) -> Population:
                return func(population, **given)
            return _operation_blueprint
        return wrapper

    return _selection
