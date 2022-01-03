import matplotlib.pyplot as plt
import numpy as np

import peal

np.random.seed(62)

pool = peal.genetics.GPPool(2, 4)


@pool.allele
def add(x: float, y: int) -> float:
    return x+y


@pool.allele
def mul(x: int, y: int) -> int:
    return x * y


@pool.allele
def one() -> int:
    return 1


@pool.allele
def two() -> int:
    return 2


@pool.allele
def threehalf() -> float:
    return 1.5


process = peal.SynchronousProcess(
    breeder=peal.Breeder(pool),
    fitness=peal.GPFitness(),
    init_size=50,
    generations=50,
    selection=peal.operations.selection.Tournament(2),
    reproduction=peal.operations.reproduction.Copy(),
    mutation=peal.operations.mutation.GPPoint(pool.configure(1, 2)),
)

tracker = peal.core.BestWorstTracker()

process.start(callbacks=[tracker])

print(tracker.best)

# - plotting -

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

ax.plot(tracker.best.fitness, label="best (normal)")
ax.set_title("Fitness")
ax.legend(loc="best")

plt.show()
