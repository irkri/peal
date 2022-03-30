import numpy as np
import matplotlib.pyplot as plt

import peal

pool = peal.gp.Pool(max_height=3)


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


@pool.allele
def iflt(x: float, y: float, r1: float, r2: float) -> float:
    return r1 if x < y else r2


pool.add_arguments({"x": float})
pool.add_terminals(np.random.random)

x_range = np.linspace(-5, 5, 100)
real_values = np.exp(-x_range**2)


@peal.fitness
def evaluate(individual: peal.Individual) -> float:
    values = np.array(
        [individual.genes[0].value(x=x) for x in x_range],
    )
    return -np.mean((values - real_values)**2)


operators = peal.operators.OperatorChain(
    peal.operators.selection.Tournament(size=3),
    peal.gp.Crossover(0.7),
    peal.gp.PointMutation(0.3),
)

strategy = peal.core.Strategy(
    operators,
    init_size=50,
    generations=100,
)

environment = peal.core.Environment(pool=pool, fitness=evaluate)

tracker = peal.callback.BestWorst()

environment.execute(strategy, callbacks=[tracker])

print(tracker.best[-1])

fig = plt.figure(figsize=(10, 5))

for i in range(20):
    axis = fig.add_subplot(4, 5, i+1)
    axis.plot(x_range, real_values)
    axis.plot(
        x_range,
        [tracker.best[(i+1)*5-1].genes[0].value(x=x) for x  in x_range]
    )
    axis.set_title(f"Generation {(i+1)*5}")

fig.tight_layout()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(tracker.best.fitness)
ax.set_title("Fitness")

plt.show()
