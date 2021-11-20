import matplotlib.pyplot as plt
import numpy as np

import peal

# individuals should have genes like this at the end of evolution
A = np.array([4, 25, 30, 40, 52, 60, 75, 80, 93, 100])


@peal.fitness
def evaluate(individual: peal.Individual) -> float:
    """Negative MSE of the individuals genes and A."""
    return -np.mean((A-individual.genes)**2)


process = peal.SynchronousProcess(
    breeder=peal.OneDimBreeder(size=10, lower=1, upper=100),
    fitness=evaluate,
    selection=peal.operations.selection.Tournament(size=3),
    mutation=peal.operations.mutation.UniformInt(
        prob=0.01,
        lowest=0,
        highest=100
    ),
    reproduction=peal.operations.reproduction.Crossover(npoints=2, prob=0.7),
)

crowded_process = peal.SynchronousProcess(
    breeder=peal.OneDimBreeder(size=10, lower=1, upper=100),
    fitness=evaluate,
    selection=peal.operations.selection.Tournament(size=3),
    mutation=peal.operations.mutation.UniformInt(
        prob=0.01,
        lowest=0,
        highest=100
    ),
    reproduction=peal.operations.reproduction.Crossover(npoints=2, prob=0.7),
    integration=peal.core.integration.CrowdedIntegration(10),
)

tracker = peal.evaluation.BestWorstTracker()
crowded_tracker = peal.evaluation.BestWorstTracker()

process.start(
    population_size=100,
    ngen=100,
    callbacks=[tracker]
)

crowded_process.start(
    population_size=100,
    ngen=100,
    callbacks=[crowded_tracker]
)

print(f"Best fitness: {tracker.best}")
print(f"Best crowded fitness: {crowded_tracker.best}")

plt.plot(tracker.best.fitness, label="best")
plt.plot(crowded_tracker.best.fitness, label="best (crowded)")
plt.legend(loc="best")
plt.show()
