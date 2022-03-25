import time
from typing import Literal, Optional

from matplotlib import pyplot as plt
import numpy as np

from peal.genetics import GenePool, GeneType, NumberPool
from peal.population import Population


class Callback:
    """Base class for all callbacks. A callback in peal is a tool that
    helps the user to supervise an evolutionary process. This way, not
    only the result of such a process can be viewed, but also other
    information that would be normally hidden and overwritten (e.g. at
    the end of generations).
    """

    def on_start(self, population: Population) -> None:
        """Will be called at the start of an evolutionary process."""

    def on_generation_start(self, population: Population) -> None:
        """Will be called at the start of each generation."""

    def on_generation_end(self, population: Population) -> None:
        """Will be called at the end of each generation."""

    def on_end(self, population: Population) -> None:
        """Will be called at the end of an evolutionary process."""


class BestWorst(Callback):
    """Class that tracks the best and worst individuals in an
    evolutionary process for each generation.

    Attributes:
        best (Population): A population containing the best individuals
            (based on fitness) from each generation. Populations are
            ordered containers and in this case the order depends on the
            generations already passed.
        worst (Population): A population containing the worst
            individuals from each generation.
    """

    def __init__(self) -> None:
        self.best: Population = Population()
        self.worst: Population = Population()

    def on_start(self, population: Population) -> None:
        self.best = Population()
        self.worst = Population()

    def on_generation_end(self, population: Population) -> None:
        self.best.integrate(max(population, key=lambda ind: ind.fitness))
        self.worst.integrate(min(population, key=lambda ind: ind.fitness))


class Diversity(Callback):
    """A callback that computes the gene diversity in a population.

    Args:
        pool (GenePool): A gene pool that is used in the evolutionary
            process to generate individuals.
    """

    def __init__(self, pool: GenePool) -> None:
        self._pool = pool
        if GeneType.CONST_SIZE not in self._pool.typing:
            raise ValueError("Diversity not available for genomes of "
                             "variable length")
        self.gene_diversity: np.ndarray = np.empty((0, 1), dtype=float)

    @property
    def diversity(self) -> np.ndarray:
        """For gene pools consisting of only categorical gene types,
        this property is the scaled average gene diversity as a float
        between 0 and 1 at each locus in a genome. For metric genomes,
        it is the mean of standard deviations across the genes of a
        population over multiple generations.
        """
        if GeneType.METRIC not in self._pool.typing:
            return (
                self._pool.size / (self._pool.size - 1)
                * self.gene_diversity.mean(axis=1)
            )
        return self.gene_diversity.mean(axis=1)

    def on_start(self, population: Population) -> None:
        self.gene_diversity = np.zeros(
            (0, population[0].genes.shape[0]),
            dtype=float
        )

    def on_generation_end(self, population: Population) -> None:
        div: np.ndarray = np.ones((population[0].genes.shape[0],))
        if GeneType.METRIC not in self._pool.typing:
            unique = set(np.hstack(list(population.genes.flatten())))
            for value in unique:
                div -= (
                    np.sum(population.genes == value, axis=0)
                    / population.size
                )**2
        else:
            div = np.std(population.genes, axis=0)
        self.gene_diversity = np.vstack([
            self.gene_diversity,
            div
        ])


class ExploreLandscape(Callback):
    """Callback that can be used on environments that evolve individuals
    with one or two genes. These genes are then drawn in a 2d or 3d plot
    together with the individuals fitness.

    Args:
        gene_pool (NumberPool): Type of gene pool in the environment
            that uses this callback.
        figax (tuple[plt.Figure, plt.Axes], optional): A matplotlib
            figure and axis that will be used for drawing. If 2d genes
            have to be plotted, the axis should be initialized with a
            ``projection="3d"`` argument. The figure will be used to
            update the canvas each time the plot changes.
        fitness_range (tuple[float, float], optional): The range of
            fitness values individuals can have. Defaults to
            ``(0.0, 1.0)``.
        sleep (float, optional): A number of seconds to wait before
            updating the plot and starting a new generation.
            Defaults to None.
    """

    def __init__(
        self,
        gene_pool: NumberPool,
        figax: Optional[tuple[plt.Figure, plt.Axes]] = None,
        fitness_range: Optional[tuple[float, float]] = None,
        sleep: Optional[float] = None,
    ) -> None:
        if not isinstance(gene_pool, NumberPool):
            raise ValueError("ExploreLandscape callback requires a NumberPool")

        if figax is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1, projection="3d")
        else:
            self.fig, self.ax = figax
        self.ax.set_xlim3d([gene_pool.lower, gene_pool.upper])
        self.ax.set_xlabel("Gene 1")
        self.ax.set_ylim3d([gene_pool.lower, gene_pool.upper])
        self.ax.set_ylabel("Gene 2")
        if fitness_range is not None:
            self.ax.set_zlim3d(fitness_range)
        self.ax.set_zlabel("Fitness")

        self._last_scattering = self.ax.scatter(
            np.empty(0),
            np.empty(0),
            np.empty(0),
            color="black",
        )
        self._sleep = sleep

    def on_generation_end(self, population: Population) -> None:
        self._last_scattering.set_color("black")
        self._last_scattering = self.ax.scatter(
            population.genes[:, 0],
            population.genes[:, 1],
            population.fitness,
            color="red",
        )

        min_ = min(population.fitness)
        max_ = max(population.fitness)
        if min_ > (old_min := self.ax.get_zlim3d()[0]):
            min_ = old_min
        if max_ < (old_max := self.ax.get_zlim3d()[1]):
            max_ = old_max
        self.ax.set_zlim3d(min_, max_)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if self._sleep is not None:
            time.sleep(self._sleep)


class VisualTracker(Callback):
    """Callback that tracks statistically important values in an
    animated plot while the evolution takes place. While this might help
    to quickly stop a process if some convergence properties seem to
    arise, it might also slow down the whole program.

    Args:
        kind (str, optional): Either 'min', 'max' or 'avg'. This option
            describes what information of the populations fitness to
            plot. Defaults to 'max'.
        figax (tuple[plt.Figure, plt.Axes], optional): A matplotlib
            figure and axis that will be used for drawing.
        fitness_range (tuple[float, float], optional): The range of
            fitness values individuals can have. Defaults to
            ``(0.0, 1.0)``.
        sleep (float, optional): A number of seconds to wait before
            updating the plot and starting a new generation.
            Defaults to None.
    """

    def __init__(
        self,
        kind: Literal["min", "max", "avg"] = "max",
        figax: Optional[tuple[plt.Figure, plt.Axes]] = None,
        fitness_range: Optional[tuple[float, float]] = None,
        sleep: Optional[float] = None,
    ) -> None:
        if figax is None:
            self.fig, self.ax = plt.subplots(1, 1)
        else:
            self.fig, self.ax = figax

        self.ax.set_xlim(0, 10)
        if fitness_range is None:
            fitness_range = (0, 1)
        self.ax.set_ylim(*fitness_range)

        self._gen = 0
        self._line, = self.ax.plot(np.empty(0), np.empty(0), marker="x")
        self._sleep = sleep
        self._kind = kind

    def on_generation_end(self, population: Population) -> None:
        self._gen += 1
        if self._gen > self.ax.get_xlim()[1]:
            self.ax.set_xlim(0, self._gen + 9)
        if self._kind == "max":
            value = max(population.fitness)
            self.ax.set_title(f"Maximal Fitness: {value:.2f}")
        elif self._kind == "avg":
            value = sum(population.fitness) / population.size
            self.ax.set_title(f"Average Fitness: {value:.2f}")
        else:
            value = min(population.fitness)
            self.ax.set_title(f"Minimal Fitness: {value:.2f}")
        if value > self.ax.get_ylim()[1]:
            self.ax.set_ylim(self.ax.get_ylim()[0], value)
        if value < self.ax.get_ylim()[0]:
            self.ax.set_ylim(value, self.ax.get_ylim()[1])
        self._line.set_data(
            np.concatenate((
                self._line.get_data()[0],
                [self._gen],
            )),
            np.concatenate((
                self._line.get_data()[1],
                [value],
            )),
        )
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        if self._sleep is not None:
            time.sleep(self._sleep)
