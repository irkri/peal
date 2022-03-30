from dataclasses import dataclass
from typing import Optional

from peal.community import Community
from peal.core.callback import Callback
from peal.core.strategy import Strategy
from peal.fitness import Fitness
from peal.genetics import GenePool
from peal.individual import Individual
from peal.operators.operator import OperatorChain
from peal.population import Population


@dataclass
class Environment:
    """An environment for the evolution of individuals and populations.
    This class is responsible for describing the creation and evaluation
    of single individuals. With this information, it executes a given
    evolutionary strategy.

    Args:
        pool (GenePool): The gene pool used for individual
            initialization.
        fitness (Fitness): The fitness to use for evaluation off
            individuals.
    """

    pool: GenePool
    fitness: Fitness

    def execute(
        self,
        population_strategy: Strategy,
        community_strategy: Optional[Strategy] = None,
        callbacks: Optional[list[Callback]] = None,
    ) -> None:
        """Performs a evolution given evolutionary strategies.

        Args:
            population_strategy (Strategy): The strategy to use for
                population evolution.
            community_strategy (Strategy, optional): The strategy to use
                for community evolution.
            callbacks (list[Callback], optional): A number of callbacks
                that are used to track information of the evolution
                following the given strategies.
        """
        if community_strategy is None:
            community_strategy = Strategy(
                OperatorChain(),
                init_size=1,
                generations=1,
            )
        callbacks = [] if callbacks is None else callbacks

        populations = Community()
        for i in range(community_strategy.init_size):
            population = Population()
            for _ in range(population_strategy.init_size):
                population.integrate(Individual(self.pool.create_genome()))
            populations.integrate(population)
            self.fitness.evaluate(populations[-1])
            for callback in callbacks:
                callback.on_start(populations[-1])

        for _ in range(community_strategy.generations):
            populations = community_strategy.operator_chain.process(
                populations,
                pool=self.pool,
            )
            self.fitness.evaluate(populations)
            for _ in range(population_strategy.generations):
                for i, parents in enumerate(populations):
                    for callback in callbacks:
                        callback.on_generation_start(parents)

                    populations[i] = (
                        population_strategy.operator_chain.process(
                            parents,
                            pool=self.pool,
                        )
                    )
                    self.fitness.evaluate(populations[i])

                    for callback in callbacks:
                        callback.on_generation_end(populations[i])
