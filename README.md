# PEAL

Python Package for Evolutionary Algorithms

## Some Notes

PEAL is a python package enabling the user to easily find solutions to optimization problems by
creating implementations of basic evolutionary algorithms.

The code is highly object-oriented.
Other packages in this field (such as [DEAP](https://github.com/DEAP/deap)) seem
to implement the algorithms in a non-pythonic way and while this coding style supplies much more
flexibility, the user also has to write more code for each new application.

PEAL aims to minimize to code a user has to write in order to implement basic algorithms.
The hard part here is figuring out how this can be achieved while also trying to
maximize the number of different methods and algorithms the user can access.
A major part of PEAL will also be an advanced type hinting structure. This automatically
increases readability and allows easier code management.

## Installation

PEAL can be installed on your local machine by using the file [setup.py](setup.py).

    >>> python setup.py install

However, it is recommended to use virtual environments together with ``poetry`` and utilize the
file [pyproject.toml](pyproject.toml).

## Documentation

The documentation of PEAL can be created by calling `make html` in the [docs](docs) folder.
This will need a few dependencies to work.
Please install the following packages using `pip` or `conda` before executing the `make` command.
- sphinx
- sphinx_rtd_theme

You can also use ``poetry`` for this by executing:

    >>> poetry install -E docs

This should create a local directory `docs/build`. Open the file `docs/build/index.html` in a
browser to access the documentation.

## Examples

Some example code on how to use PEAL can be found in the [examples](examples) folder.
