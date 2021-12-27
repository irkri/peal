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


process = peal.StrategyProcess(
    breeder=peal.Breeder(gene_pool=pool),
    fitness=evaluate,
    mutation=peal.operations.mutation.UniformInt(
        prob=0.1,
        lowest=0,
        highest=100,
    ),
    signature="3/1,5(2/2,14)^10"
)

tracker = peal.evaluation.BestWorstTracker()
statistics = peal.evaluation.DiversityStatistics(allele=np.arange(1, 101))
process.start(
    ngen=5,
    callbacks=[tracker, statistics]
)

print(tracker.best)

# - plotting -

fig, ax = plt.subplots(2, 1, figsize=(10, 8))

ax[0].plot(tracker.best.fitness, label="best (normal)")
ax[0].set_title("Fitness")
ax[0].legend(loc="best")

ax[1].plot(statistics.diversity, label="normal")
ax[1].set_title("Diversity")
ax[1].legend(loc="best")

plt.show()
