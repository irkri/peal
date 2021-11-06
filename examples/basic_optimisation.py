import numpy as np
import peal

population = peal.Population()
for i in range(100):
    ind = peal.Individual(np.random.randint(0, 101, size=(10,)))
    population.populate(ind)

# individuals should have genes like this array at the end of evolution
A = np.array([4, 25, 30, 40, 52, 60, 75, 80, 93, 100])

def evaluate(individual: peal.Individual) -> float:
    return -np.mean((A-individual.genes)**2)


fitness = peal.Fitness(evaluate)

env = peal.Environment(fitness)
env.push(population)

fitness.evaluate(env._population)

for i in env._population:
    print(i.fitness, end=" ")

print("\n\n")

for i in env._population:
    print(i.genes)

env.use(peal.operations.mutation.uniform_int(prob=0.05, lowest=0, highest=100))
env.use(peal.operations.reproduction.crossover(npoints=2, prob=0.7))
env.use(peal.operations.selection.tournament(size=3))

tracker = peal.evaluation.FitnessTracker()

env.evolve(ngen=50, callbacks=[tracker])

fitness.evaluate(env._population)

for i in env._population:
    print(i.fitness, end=" ")

print("\n\n")

for i in env._population:
    print(i.genes)

print(f"Best fitness: {tracker.best}")
print(f"Average fitness: {tracker.average}")
print(f"Worst fitness: {tracker.worst}")

import matplotlib.pyplot as plt

plt.plot(tracker.best, label="best")
plt.plot(tracker.average, label="average")
plt.plot(tracker.worst, label="worst")
plt.legend(loc="best")
plt.show()
