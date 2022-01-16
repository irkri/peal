import numpy as np
import matplotlib.pyplot as plt

import peal

pool = peal.genetics.GPPool(min_depth=1, max_depth=3)


@pool.allele
def add(x: float, y: float) -> float:
    return x + y


@pool.allele
def mul(x: float, y: float) -> float:
    return x * y


@pool.allele
def sin(x: float) -> float:
    return np.sin(x)


@pool.allele
def cos(x: float) -> float:
    return np.cos(x)


pool.add_arguments({"x": float})


X = np.linspace(-5, 5, 100)
args = [{"x": X[i]} for i in range(len(X))]


def evaluate(values: list[float]) -> float:
    return -np.sum(np.abs(values-np.exp(-X**2)))


strategy = peal.core.Strategy(
    generations=100,
    init_individuals=50,
    reproduction=peal.operations.reproduction.Copy(),
    mutation=peal.operations.mutation.GPPoint(
        gene_pool=pool,
        min_height=1,
        max_height=3,
        prob=0.1,
    ),
    selection=peal.operations.selection.Tournament(size=3),
)

environment = peal.core.Environment(
    breeder=peal.Breeder(pool),
    fitness=peal.GPFitness(arguments=args, evaluation=evaluate),
)

tracker = peal.callback.BestWorst()

environment.execute(strategy, callbacks=[tracker])

print(tracker.best[-1])

fig, ax = plt.subplots(2, 1)
# ax2 = ax.twinx()

ax[0].plot(tracker.best.fitness, color="blue")
ax[0].set_title("Fitness/Diversity")
ax[1].plot(X, np.exp(-X**2))
ax[1].plot(
    X,
    [peal.evaluation.gp_evaluate(tracker.best[-1], argset) for argset in args]
)

plt.show()
