import matplotlib.pyplot as plt
import numpy as np

import peal

# individuals should have genes like this at the end of evolution
A = np.array([4, 74, 43, 22, 100])
B = np.array([8, 34, 65, 21, 99])


@peal.fitness
def evaluate(individual: peal.Individual) -> float:
    """Negative MSE of the individuals genes and A."""
    return -np.mean((A-individual.genes)**2) - np.mean((B-individual.genes)**2)


process = peal.StrategyProcess(
    breeder=peal.IntegerBreeder(size=A.size, lower=1, upper=100),
    fitness=evaluate,
    mutation=peal.operations.mutation.UniformInt(
        prob=0.1,
        lowest=0,
        highest=100,
    ),
    signature="(5/2,10)"
)

tracker = peal.evaluation.BestWorstTracker()
statistics = peal.evaluation.DiversityStatistics(allele=np.arange(1, 101))
process.start(
    ngen=100,
    callbacks=[tracker, statistics]
)

# - plotting -

fig, ax = plt.subplots(2, 1, figsize=(10, 8))

ax[0].plot(tracker.best.fitness, label="best (normal)")
ax[0].set_title("Fitness")
ax[0].legend(loc="best")

ax[1].plot(statistics.diversity, label="normal")
ax[1].set_title("Diversity")
ax[1].legend(loc="best")

plt.show()
