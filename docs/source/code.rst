Getting Started
===============
PEAL is a python package supplying an easy way to create implementations of solutions to
optimization problems by using basic evolutionary algorithms.
The object-oriented programming style that was used to build PEAL allows for various types of
adjustments to the code that is already written.

I started this project as a collection of classes that help me quickly finish some homework tasks
for university.
Also, I felt like other packages in this field (such as `DEAP <https://github.com/DEAP/deap>`_)
seem to implement the algorithms in a non-pythonic way without using an here naturally suited
object-oriented approach. Of course, these package tend to have a much higher variety of methods
and **PEAL** should (for now) not be seen as a replacement of mentioned advanced tools.

To get a basic understanding of the structure of the package, I encourage you to dig through
some of the topics listed below (in the order they appear). This documentation was quickly created
to format code docstring in a better way and add some comments to them. Future updates will expand
these pages and hopefully provide more detail.

Choosing a Process
==================
.. toctree::
   :maxdepth: 4
   :glob:

   packages/core

Populations and Individuals
===========================
.. toctree::
   :maxdepth: 4
   :glob:

   packages/population

Your Part: Evaluation and Supervision
=====================================
.. toctree::
   :maxdepth: 4
   :glob:

   packages/evaluation

Customizing a Process: Evolutionary Operators
=============================================
.. toctree::
   :maxdepth: 4
   :glob:

   packages/operations/operator
   packages/operations/*
   packages/integration
