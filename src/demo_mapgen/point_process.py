from typing import Union

import numpy as np


class IntensityFunction:

    @property
    def rate(self) -> float:
        raise NotImplementedError

    def is_accepted(self, value: tuple[int, ...], threshold: float) -> bool:
        raise NotImplementedError


class PointProcess(object):
    def __init__(self, rate: Union[int, float, IntensityFunction], size: tuple[int, ...], seed: int) -> None:
        if not isinstance(size, tuple) or any(not isinstance(dim, int) for dim in size):
            raise TypeError("Argument 'size' should be a tuple of integer numbers, not '%s'" % type(size).__name__)
        if any(dim <= 0 for dim in size):
            raise ValueError("Argument 'size' should be positive")
        if not isinstance(rate, (int, float, IntensityFunction)):
            raise TypeError("Argument 'rate' should be number or %s, not '%s'" % (
                IntensityFunction.__name__, type(rate).__name__))
        if ((isinstance(rate, (int, float)) and rate <= 0) or
                (isinstance(rate, IntensityFunction) and rate.rate <= 0)):
            raise ValueError("Argument 'rate' should be positive")

        if not isinstance(seed, int):
            raise TypeError("Argument 'seed' should be integer number, not '%s'" % type(seed).__name__)

        self.rate = rate

        self.size = size
        self._rng = np.random.default_rng(seed)

    def __iter__(self):
        raise NotImplementedError


class MarkovChainMonteCarlo(PointProcess):

    def __iter__(self):
        n = self._rng.poisson(self.rate.rate if isinstance(self.rate, IntensityFunction) else self.rate)
        count = 0
        while count <= n:
            candidate = tuple(self._rng.integers([0] * len(self.size), self.size))
            if isinstance(self.rate, IntensityFunction):
                # non-homogeneous
                d = self._rng.uniform(0, 1)
                if self.rate.is_accepted(candidate, d):
                    count += 1
                    yield candidate
            else:
                # homogeneous
                count += 1
                yield candidate


class SpatialPoissonPointProcess(PointProcess):

    def __iter__(self):
        n = self._rng.poisson(self.rate.rate if isinstance(self.rate, IntensityFunction) else self.rate)
        for _ in range(n):
            candidate = tuple(self._rng.integers([0] * len(self.size), self.size))
            if isinstance(self.rate, IntensityFunction):
                # non-homogeneous
                d = self._rng.uniform(0, 1)
                if self.rate.is_accepted(candidate, d):
                    yield candidate
            else:
                # homogeneous
                yield candidate
