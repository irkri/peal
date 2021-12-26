# PEAL

Python Package for Evolutionary Algorithms

## Some Notes

__PEAL__ is a python package enabling the user to easily create implementations of solutions to
optimization problems by using basic evolutionary algorithms while also supplying an intuitive way
of creating new methods.

I started this project as a collection of classes that help me quickly finish some homework tasks
for university.
Also, I felt like other packages in this field (such as [DEAP](https://github.com/DEAP/deap))
seem to implement the algorithms in a non-pythonic way without using an here naturally suited
object-oriented approach. Of course, these package tend to have a much higher variety of methods
and __PEAL__ should (for now) not be seen as a replacement of mentioned advanced tools.

If you have tips, ideas or possible changes on the structure of __PEAL__ you want me to know about,
do not hesitate to contact me. I'm still trying to figure out a lot of things about evolutionary
algorithms as well as python package development.

## Installation

__PEAL__ can be installed on your local machine by using the file [setup.py](setup.py).
```
  $ python setup.py install
```

## Documentation

The documentation of __PEAL__ can be created by calling `make html` in the [docs](docs) folder. This will need a few dependencies to work. Please install the following packages using `pip` or `conda` before executing the `make` command.
- sphinx
- sphinx_rtd_theme

This should create a local directory `docs/build`. Open the file `docs/build/index.html` in a browser to access the documentation.

## Examples

Some example code on how to use __PEAL__ can be found in the [examples](examples) folder.
