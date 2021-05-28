from math import ceil, log, exp
from typing import Union

from PIL import Image, ImageStat, ImageChops

from .point_process import IntensityFunction


# Abstract classes


class MarkovChainMonteCarloPotentialFunction(object):
    def get(self, value: tuple[int, int, int]) -> float:
        raise NotImplementedError

    def update(self, value: tuple[int, int, int]) -> None:
        raise NotImplementedError


class SpatialPoissonPointProcessPotentialFunction(MarkovChainMonteCarloPotentialFunction):
    def get(self, value):
        raise NotImplementedError

    def update(self, value):
        raise NotImplementedError

    @property
    def expected(self) -> float:
        raise NotImplementedError


class MarkovChainMonteCarloCompositeFunction(object):
    def get(self, value: tuple[int, int, int], kernel: float, potential: float) -> float:
        raise NotImplementedError


class SpatialPoissonPointProcessCompositeFunction(MarkovChainMonteCarloCompositeFunction):
    def get(self, value, kernel, potential):
        raise NotImplementedError

    @property
    def expected(self) -> float:
        raise NotImplementedError


# Implementation


class AdaptivePotentialFunction(MarkovChainMonteCarloPotentialFunction):
    _monopole_potential_cache = {}

    def __init__(self, size: int, city_sizes: int) -> None:
        self._city_sizes = city_sizes

        self._potential_image = Image.new('L', (size, size), 0)

    @classmethod
    def clear_cache(cls) -> None:
        cls._monopole_potential_cache.clear()

    @property
    def potential_map(self) -> Image.Image:
        return self._potential_image

    def get(self, value):
        x, y, z = value
        pixel = (255 - self._potential_image.getpixel((x, y))) / 255
        return pixel

    def update(self, value):
        x, y, z = value
        chop = Image.new('L', self._potential_image.size, 0)
        potential_image = self._get_monopole_potential(z + 1)
        s, s = potential_image.size
        x0 = y0 = s // 2
        chop.paste(potential_image, (x - x0, y - y0))
        self._potential_image = ImageChops.add(self._potential_image, chop)

    def _get_monopole_potential(self, r: int) -> Image.Image:
        if r not in self._monopole_potential_cache:
            p = 1 / 3
            s = ceil(2 ** .5 * r * (log(510 * r / (self._city_sizes + 1))) ** (.5 / p))
            potential_image = Image.frombytes(
                'L', (2 * s, 2 * s),
                bytes(round(255 * r / (self._city_sizes + 1) * exp(-((x ** 2 + y ** 2) / 2 / r ** 2) ** p))
                      for x in range(-s, s) for y in range(-s, s))
            )
            self._monopole_potential_cache[r] = potential_image
        return self._monopole_potential_cache[r]


class ExponentialZCompositeFunction(MarkovChainMonteCarloCompositeFunction):
    def get(self, value, kernel, potential):
        rate = (potential * kernel) ** (value[2] + 1)
        return rate


class MarkovChainMonteCarloIntensityFunction(IntensityFunction):

    def __init__(self, rate: Union[int, float], kernel_image: Image.Image,
                 potential_func: MarkovChainMonteCarloPotentialFunction,
                 composite_func: MarkovChainMonteCarloCompositeFunction) -> None:
        super().__init__()
        self._rate = rate
        self._potential_function = potential_func
        self._composite_function = composite_func
        self._kernel_image = kernel_image

        self._mean = (255 - ImageStat.Stat(self._kernel_image).mean.pop()) / 255

    @property
    def rate(self):
        return self._rate

    def is_accepted(self, value: tuple[int, int, int], threshold: float) -> bool:
        x, y, z = value
        p = self._potential_function.get(value)
        k = (255 - self._kernel_image.getpixel((x, y))) / 255
        if threshold <= self._composite_function.get(value, k, p):
            self._potential_function.update(value)
            return True
        return False


class SpatialPoissonPointProcessIntensityFunction(MarkovChainMonteCarloIntensityFunction):
    def __init__(self, rate: Union[int, float], kernel_image: Image.Image,
                 potential_func: SpatialPoissonPointProcessPotentialFunction,
                 composite_func: SpatialPoissonPointProcessCompositeFunction) -> None:
        super().__init__(rate, kernel_image, potential_func, composite_func)

    @property
    def rate(self):
        self._potential_function: SpatialPoissonPointProcessPotentialFunction
        self._composite_function: SpatialPoissonPointProcessCompositeFunction
        return self._rate / self._composite_function.expected / self._potential_function.expected
