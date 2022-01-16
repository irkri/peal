from dataclasses import dataclass

from peal.breeding import Breeder
from peal.community import Community
from peal.core.callback import Callback
from peal.core.strategy import Strategy
from peal.fitness import Fitness


@dataclass
class Environment:

    breeder: Breeder
    fitness: Fitness

    def execute(self, strategy: Strategy, callbacks: list[Callback]) -> None:
        """Executes the given evolutionary strategy."""
        callbacks = [] if callbacks is None else callbacks

        parent_populations = Community()
        for i in range(strategy.init_populations):
            parent_populations.integrate(
                self.breeder.breed(strategy.init_individuals)
            )
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
