from math import ceil, exp, log
from typing import Union

from PIL import Image, ImageStat, ImageChops

from .common import PointType
from .point_process import IntensityFunction


# Abstract classes


class MarkovChainMonteCarloPotentialFunction:
    def get(self, value: PointType) -> float:
        raise NotImplementedError

    def update(self, value: PointType) -> None:
        raise NotImplementedError


class SpatialPoissonPointProcessPotentialFunction(MarkovChainMonteCarloPotentialFunction):
    def get(self, value):
        raise NotImplementedError

    def update(self, value):
        raise NotImplementedError

    def get_expected(self) -> float:
        """It is the expected value of the potential function."""

        raise NotImplementedError


class MarkovChainMonteCarloCompositeFunction:
    def get(self, value: PointType, kernel: float, potential: float) -> float:
        raise NotImplementedError


class SpatialPoissonPointProcessCompositeFunction(MarkovChainMonteCarloCompositeFunction):
    """It is an abstract function for implementing composite functions.

    It is used in :class:`~.point_process.SpatialPoissonPointProcess` method to generate a point
    process. It is differ from :class:`.MarkovChainMonteCarloCompositeFunction` by having an
    :meth:`.get_expected` method (see: there).
    """

    def get(self, value, kernel, potential):
        raise NotImplementedError

    def get_expected(self, expected_kernel: float, expected_potential: float) -> float:
        """It should be the expected value of the implemented arithmetics.

        This allows :class:`.SpatialPoissonPointProcessIntensityFunction` to scale
        :attr:`~.SpatialPoissonPointProcessIntensityFunction.rate` so that the generated point
        process remain *similar* to a Poison point process.

        :param expected_kernel: It is the expected value of the kernel function.
        :type expected_kernel: :class:`float`
        :param expected_potential: It is the expected value of the potential function.
        :type expected_potential: :class:`float`
        :return: It is the expected value of the composite function.
        :rtype: :class:`float`
        """

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
        x, y, _z = value
        pixel = (255 - self._potential_image.getpixel((x, y))) / 255
        return pixel

    def update(self, value):
        x, y, z = value
        chop = Image.new('L', self._potential_image.size, 0)
        potential_image = self._get_monopole_potential(z + 1)
        x0 = y0 = potential_image.size[0] // 2
        chop.paste(potential_image, (x - x0, y - y0))
        self._potential_image = ImageChops.add(self._potential_image, chop)

    def _get_monopole_potential(self, radius: int) -> Image.Image:
        if radius not in self._monopole_potential_cache:
            p = 1 / 3
            size = ceil(2 ** .5 * radius * (log(510 * radius / (self._city_sizes + 1))) ** (.5 / p))

            def super_gauss(x: int, y: int, sigma: int, power: float):
                a = 255 * sigma / (self._city_sizes + 1)
                return a * exp(-((x ** 2 + y ** 2) / 2 / sigma ** 2) ** power)

            potential_image = Image.frombytes(
                'L', (2 * size, 2 * size),
                bytes(round(super_gauss(x, y, radius, p))
                      for x in range(-size, size) for y in range(-size, size))
            )
            self._monopole_potential_cache[radius] = potential_image
        return self._monopole_potential_cache[radius]


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

    @property
    def rate(self):
        return self._rate

    def is_accepted(self, value: PointType, threshold: float) -> bool:
        x, y, _z = value
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

        self._mean = (255 - ImageStat.Stat(self._kernel_image).mean.pop()) / 255

    @property
    def rate(self):
        """It is the expected number of points generated.

        :return: It is calculated based on the value set by `rate` during initialization, and the
            :attr:`~.SpatialPoissonPointProcessCompositeFunction.get_expected` attribute of
            `composite_func`.
        :rtype: :class:`float`
        """
        self._potential_function: SpatialPoissonPointProcessPotentialFunction
        self._composite_function: SpatialPoissonPointProcessCompositeFunction
        return self._rate / self._composite_function.get_expected(
            self._mean, self._potential_function.get_expected())
