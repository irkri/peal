from typing import Iterable, Optional, Sequence, SupportsIndex, Union, overload

from peal.population import Population


class Community:
    """An iterable container for :class:`~peal.population.Population`
    objects.

    Args:
        populations (Population | Iterable[Population]): One or more
            populations to add.
    """

    def __init__(
        self,
        populations: Optional[Union[Population, Iterable[Population]]] = None,
    ):
        self._iter_id = -1
        self._populations: list[Population] = []
        if populations is not None:
            self.integrate(populations)

    @property
    def size(self) -> int:
        """Returns the number of populations in the community."""
        return len(self._populations)

    def integrate(
        self,
        populations: Union[Population, Iterable[Population]],
    ) -> None:
        """Integrate populations into the community.

        Args:
            populations (Population | Iterable[Population]): One or
                multiple populations.
        """
        if isinstance(populations, Population):
            self._populations.append(populations)
        elif isinstance(populations, Iterable):
            for population in populations:
                if not isinstance(population, Population):
                    raise TypeError("Can only append populations to a "
                                    f"community, got {type(population)}")
                self._populations.append(population)
        else:
            raise TypeError("Can only append populations to a population, "
                            f"got {type(populations)}")

    def copy(self) -> "Community":
        """Returns a shallow copy of this community."""
        return Community(self)

    def deepcopy(self) -> "Community":
        """Returns a deep copy of this community that is also copying
        all populations.
        """
        return Community(tuple(pop.deepcopy() for pop in self._populations))

    def __iter__(self) -> "Community":
        self._iter_id = -1
        return self

    def __next__(self) -> Population:
        if self._iter_id == len(self._populations) - 1:
            raise StopIteration
        self._iter_id += 1
        return self._populations[self._iter_id]

    @overload
    def __getitem__(self, key: SupportsIndex) -> Population:
        ...

    @overload
    def __getitem__(self, key: slice) -> "Community":
        ...

    @overload
    def __getitem__(self, key: Sequence[int]) -> "Community":
        ...

    def __getitem__(
        self,
        key: Union[SupportsIndex, slice, Sequence[int]],
    ) -> Union["Community", Population]:
        if isinstance(key, slice):
            return Community(self._populations[key])
        elif isinstance(key, Sequence):
            return Community([self._populations[i] for i in key])
        return self._populations.__getitem__(key)

    @overload
    def __setitem__(self, key: SupportsIndex, value: Population) -> None:
        ...

    @overload
    def __setitem__(self, key: slice, value: Iterable[Population]) -> None:
        ...

    def __setitem__(self, *args) -> None:
        self._populations.__setitem__(*args)

    def __repr__(self) -> str:
        return f"Community(size={self.size})"
