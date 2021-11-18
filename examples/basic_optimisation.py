import matplotlib.pyplot as plt
import numpy as np

import peal


@peal.breeder
def create() -> peal.Individual:
    return peal.Individual(np.random.randint(1, 101, size=(10,)))


# individuals should have genes like this array at the end of evolution
A = np.array([4, 25, 30, 40, 52, 60, 75, 80, 93, 100])


@peal.fitness
def evaluate(individual: peal.Individual) -> float:
    return -np.mean((A-individual.genes)**2)


process = peal.SynchronousProcess(
    breeder=create,
    fitness=evaluate,
    selection=peal.operations.selection.Tournament(size=3),
    mutation=peal.operations.mutation.UniformInt(
        prob=0.01,
        lowest=0,
        highest=100
    ),
    reproduction=peal.operations.reproduction.Crossover(npoints=2, prob=0.7)
)

tracker = peal.evaluation.BestWorstTracker()

process.start(
    population_size=100,
    ngen=100,
    callbacks=[tracker]
)

print(f"Best fitness: {tracker.best}")
print(f"Worst fitness: {tracker.worst}")

plt.plot(tracker.best.fitness, label="best")
plt.plot(tracker.worst.fitness, label="worst")
plt.legend(loc="best")
plt.show()
