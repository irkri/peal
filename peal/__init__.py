import importlib.metadata

__version__ = importlib.metadata.version("peal")

from peal import core
from peal.core import callback

from peal import operators

from peal.fitness import fitness, Fitness

from peal import genetics
from peal import gp
from peal.individual import Individual
from peal.population import Population
from peal.community import Community
