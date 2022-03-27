import numpy as np
import matplotlib.pyplot as plt

import peal

pool = peal.gp.Pool(min_depth=1, max_depth=3)


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
pool.add_terminals(np.random.random)


X = np.linspace(-5, 5, 100)
args = [{"x": X[i]} for i in range(len(X))]


def evaluate(values: list[float]) -> float:
    return - np.mean((values-np.exp(-X**2))**2)


strategy = peal.core.Strategy(
    init_individuals=50,
    generations=100,
    reproduction=peal.operators.reproduction.Copy(),
    mutation=peal.gp.PointMutation(
        gene_pool=pool,
        min_height=1,
        max_height=3,
        prob=0.3,
    ),
    selection=peal.operators.selection.Tournament(size=3),
)

environment = peal.core.Environment(
    pool=pool,
    fitness=peal.gp.Fitness(arguments=args, evaluation=evaluate),
)

tracker = peal.callback.BestWorst()

environment.execute(strategy, callbacks=[tracker])

print(tracker.best[-1])

fig = plt.figure(figsize=(10, 5))

for i in range(20):
    axis = fig.add_subplot(4, 5, i+1)
    axis.plot(X, np.exp(-X**2))
    axis.plot(
        X,
        [peal.gp.evaluate(tracker.best[(i+1)*5-1], argset) for argset in args]
    )
    axis.set_title(f"Generation {(i+1)*5}")

fig.tight_layout()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(tracker.best.fitness)
ax.set_title("Fitness")

plt.show()
