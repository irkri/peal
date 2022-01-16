import matplotlib.pyplot as plt
import numpy as np

import peal

A = np.array([4, 74, 43, 23, 0])

pool = peal.genetics.IntegerPool(shape=A.size, lower=0, upper=101)


@peal.fitness
def evaluate(individual: peal.Individual) -> float:
    """Negative MSE comparing individuals genes to A."""
    return -np.mean((A - individual.genes)**2)


process = peal.SynchronousProcess(
    breeder=peal.Breeder(pool),
    fitness=evaluate,
    init_size=100,
    generations=100,
    selection=peal.operations.selection.Tournament(size=4),
    mutation=peal.operations.mutation.UniformInt(
        prob=0.1,
        lowest=0,
        highest=100,
    ),
    reproduction=peal.operations.reproduction.Crossover(
        npoints=1,
        probability=0.7,
    ),
    integration=peal.operations.integration.Crowded(10),
)

tracker = peal.callback.BestWorst()
statistics = peal.callback.Diversity(pool=pool)

process.start(callbacks=[tracker, statistics])

print(tracker.best)

# - plotting -

fig, ax = plt.subplots(2, 1, figsize=(10, 8))

ax[0].plot(tracker.best.fitness)
ax[0].set_title("Fitness")

ax[1].plot(statistics.diversity)
ax[1].set_title("Diversity")

plt.show()
