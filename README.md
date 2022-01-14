# PEAL

Python Package for Evolutionary Algorithms

## Some Notes

PEAL is a python package enabling the user to easily find solutions to optimization problems by
creating implementations of basic evolutionary algorithms.

This project started as a collection of classes that help me finish some homework tasks for
university. Also, I felt like other packages in this field (such as
[DEAP](https://github.com/DEAP/deap)) seem to implement the algorithms in a non-pythonic way
without using an here naturally suited object-oriented approach.
This implementation style supplies a much higher variety of things that can be done with these
packages with the drawback of having to write more code for each new application.

PEAL wants the user to write only as much code as really is needed to get the given problem
solved. The hard part here will be figuring out how this can be achieved while also trying to
maximize the number of different methods and algorithms one can access.
A major part of PEAL will also be an advanced type hinting structure. This automatically
increases readability and helps understanding the code.

## Installation

PEAL can be installed on your local machine by using the file [setup.py](setup.py).
```
  $ python setup.py install
```

## Documentation

The documentation of PEAL can be created by calling `make html` in the [docs](docs) folder.
This will need a few dependencies to work.
Please install the following packages using `pip` or `conda` before executing the `make` command.
- sphinx
- sphinx_rtd_theme

This should create a local directory `docs/build`. Open the file `docs/build/index.html` in a
browser to access the documentation.

## Examples

Some example code on how to use PEAL can be found in the [examples](examples) folder.
