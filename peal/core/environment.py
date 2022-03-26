from dataclasses import dataclass

from peal.community import Community
from peal.core.callback import Callback
from peal.core.strategy import Strategy
from peal.fitness import Fitness
from peal.genetics import GenePool
from peal.individual import Individual
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

    def execute(self, strategy: Strategy, callbacks: list[Callback]) -> None:
        """Executes the given evolutionary strategy.

        Args:
            strategy (Strategy): The strategy to execute.
            callbacks (list[Callback]): A number of callbacks that are
                used to track information of the evolution following the
                given strategy.
        """
        callbacks = [] if callbacks is None else callbacks

        parent_populations = Community()
        for i in range(strategy.init_populations):
            population = Population()
            for _ in range(strategy.init_individuals):
                population.integrate(Individual(self.pool.create_genome()))
            parent_populations.integrate(population)
            if strategy.select_parent_populations:
                self.fitness.evaluate(parent_populations[-1])
            for callback in callbacks:
                callback.on_start(parent_populations[-1])

        for _ in range(strategy.population_generations):
            offspring_populations = strategy.population_reproduction.process(
                parent_populations
            )

            for _ in range(strategy.generations):
                for i, parents in enumerate(offspring_populations):
                    for callback in callbacks:
                        callback.on_generation_start(parents)

                    offspring = strategy.reproduction.process(parents)
                    offspring = strategy.mutation.process(offspring)
                    self.fitness.evaluate(offspring)
                    offspring, = strategy.integration.process(
                        Community((offspring, parents))
                    )
                    offspring_populations[i] = strategy.selection.process(
                        offspring
                    )

                    for callback in callbacks:
                        callback.on_generation_end(offspring_populations[i])

            if strategy.select_parent_populations:
                for population in parent_populations:
                    offspring_populations.integrate(population)
            parent_populations = strategy.population_selection.process(
                offspring_populations
            )
