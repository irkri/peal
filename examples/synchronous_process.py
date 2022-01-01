import matplotlib.pyplot as plt
import numpy as np

import peal

A = np.array([4, 74, 43, 23, 0])
B = np.array([8, 34, 65, 21, 100])

pool = peal.population.IntegerPool(shape=A.size, lower=0, upper=101)


@peal.fitness
def evaluate(individual: peal.Individual) -> float:
    """Negative MSE comparing individuals genes to A and B."""
    return -np.mean((A-individual.genes)**2) - np.mean((B-individual.genes)**2)


process = peal.SynchronousProcess(
    breeder=peal.Breeder(pool),
    fitness=evaluate,
    init_size=100,
    generations=100,
    selection=peal.operations.selection.Tournament(size=3),
    mutation=peal.operations.mutation.UniformInt(
        prob=0.01,
        lowest=0,
        highest=100,
    ),
    reproduction=peal.operations.reproduction.Crossover(
        npoints=1,
        probability=0.7
    ),
)
crowded_process = peal.SynchronousProcess(
    breeder=peal.Breeder(pool),
    fitness=evaluate,
    init_size=100,
    generations=100,
    selection=peal.operations.selection.Tournament(size=3),
    mutation=peal.operations.mutation.UniformInt(
        prob=0.01,
        lowest=0,
        highest=100,
    ),
    reproduction=peal.operations.reproduction.Crossover(
        npoints=1,
        probability=0.7
    ),
    integration=peal.population.CrowdedIntegration(10),
)

tracker = peal.evaluation.BestWorstTracker()
crowded_tracker = peal.evaluation.BestWorstTracker()
statistics = peal.evaluation.DiversityStatistics(allele=np.arange(1, 101))
crowded_statistics = peal.evaluation.DiversityStatistics(
    allele=np.arange(1, 101)
)

process.start(callbacks=[tracker, statistics])
crowded_process.start(callbacks=[crowded_tracker, crowded_statistics])

print(tracker.best)

# - plotting -

fig, ax = plt.subplots(2, 1, figsize=(10, 8))

ax[0].plot(tracker.best.fitness, label="best (normal)")
ax[0].plot(crowded_tracker.best.fitness, label="best (crowded)")
ax[0].set_title("Fitness")
ax[0].legend(loc="best")

ax[1].plot(statistics.diversity, label="normal")
ax[1].plot(crowded_statistics.diversity, label="crowded")
ax[1].set_title("Diversity")
ax[1].legend(loc="best")

plt.show()
