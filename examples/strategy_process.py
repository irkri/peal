import matplotlib.pyplot as plt
import numpy as np

import peal

A = np.array([4, 74, 43, 23, 0])

pool = peal.genetics.NumberPool(shape=A.size, lower=0, upper=101)


@peal.fitness
def evaluate(individual: peal.Individual) -> float:
    """Negative MSE comparing individuals genes to A."""
    return - np.mean((A - individual.genes)**2)


strategy = peal.core.Strategy.from_string(
    "1/1,2(2/1,10)^10",
    population_generations=10,
)

environment = peal.core.Environment(
    pool=pool,
    fitness=evaluate,
)

tracker = peal.callback.BestWorst()
statistics = peal.callback.Diversity(pool=pool)
environment.execute(strategy, callbacks=[tracker, statistics])

print(tracker.best)

# - plotting -

fig = plt.figure(figsize=(10, 8))

axis1 = fig.add_subplot(3, 1, 1)
axis1.plot(tracker.best.fitness, label="best (normal)")
axis1.set_title("Fitness")
axis1.legend(loc="best")

axis2 = fig.add_subplot(3, 1, 2)
axis2.plot(statistics.diversity, label="normal")
axis2.set_title("Diversity")
axis2.legend(loc="best")

axis3 = fig.add_subplot(3, 1, 3)
axis3.plot([indiv.hidden_genes[0] for indiv in tracker.best], label="normal")
axis3.set_title("Mutation step size")
axis3.legend(loc="best")

plt.show()
