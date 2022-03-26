import matplotlib.pyplot as plt
import numpy as np

import peal
from visual_callbacks import ExploreLandscape, VisualTracker

pool = peal.genetics.NumberPool(length=2, lower=-5, upper=5)


@peal.fitness
def evaluate(individual: peal.Individual) -> float:
    x, y = individual.genes
    return - (x**2 + y - 11)**2 - (x + y**2 - 7)**2


strategy = peal.core.Strategy.from_string(
    string="(10/10,20)^20",
    population_generations=1,
)

environment = peal.core.Environment(
    pool=pool,
    fitness=evaluate,
)

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax2 = fig.add_subplot(1, 2, 2)
explorer = ExploreLandscape(
    gene_pool=pool,
    figax=(fig, ax1),
)
tracker = VisualTracker(
    kind="max",
    figax=(fig, ax2),
)

y = x = np.linspace(-5, 5, 500)
x, y = np.meshgrid(x, y)

z = - (x**2 + y - 11)**2 - (x + y**2 - 7)**2

explorer.ax.plot_surface(x, y, z, cmap="plasma", alpha=0.8)

plt.ion()
plt.show()
environment.execute(strategy=strategy, callbacks=[explorer, tracker])

plt.pause(5)
