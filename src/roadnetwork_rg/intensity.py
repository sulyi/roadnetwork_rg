"""Various implementation of :class:`.IntensityFunction`

It provides some functions to be used for generating non-homogenous a
:class:`~.point_process.PointProcess`, (either using
:class:`~.point_process.SpatialPoissonPointProcess` or
:class:`~.point_process.MarkovChainMonteCarlo` method) by implementing the interface used for
thinning points offered by the point process as candidates.
"""

from math import ceil, exp, log
from typing import Union

from PIL import Image, ImageStat, ImageChops

from .common import PointType
from .point_process import IntensityFunction


# Abstract classes


class MarkovChainMonteCarloPotentialFunction:
    """It is an abstract class for implementing potential functions.

    A potential function describes the point-wise interaction between the generated points.
    It provides an interface used by :class:`MarkovChainMonteCarloIntensityFunction`,
    (see also: there).
    """

    def get(self, value: PointType) -> float:
        """It should implemented the function arithmetics.

        :param value: It is the point where the function is computed.
        :type value: :attr`.PointType`
        :return: It is the calculated value.
        :rtype: :class:`float`
        """

        raise NotImplementedError

    def update(self, value: PointType) -> None:
        """It updates the function based on whether the candidate is accepted or rejected.

        :param value: It is the candidate where the potential function being updated.
        :type value: :attr:`.PointType`
        """

        raise NotImplementedError


class SpatialPoissonPointProcessPotentialFunction(MarkovChainMonteCarloPotentialFunction):
    """It is an abstract class for implementing potential functions.

    See also: :class:`MarkovChainMonteCarloPotentialFunction`.
    It also provides an interface used by :class:`SpatialPoissonPointProcessIntensityFunction`
    """

    def get(self, value):
        raise NotImplementedError

    def update(self, value):
        raise NotImplementedError

    @property
    def expected(self) -> float:
        """It is the expected value of the potential function."""

        raise NotImplementedError


class MarkovChainMonteCarloCompositeFunction:
    """It is an abstract function for implementing composite functions.

    It is used by :class:`.MarkovChainMonteCarloIntensityFunction` to calculate value used for
    thinning a point process. A composite function takes two other function a kernel and a potential
    function and computes a value based on them according the implemented arithmetics.
    """

    def get(self, value: PointType, kernel: float, potential: float) -> float:
        """The arithmetics used should be implemented by this.

        :param value: It is the point at where the function is calculated.
        :type value: :attr:`.PointType`
        :param kernel: It is the value of the kernel function at the point `value`.
        :type kernel: :class:`float`
        :param potential: It is the value of the potential function at the point `value`.
        :type potential: :class:`float`
        :returns: It is the result of computation.
        :rtype: :class:`float`
        """

        raise NotImplementedError


class SpatialPoissonPointProcessCompositeFunction(MarkovChainMonteCarloCompositeFunction):
    """It is an abstract function for implementing composite functions.

    It is used in :class:`~.point_process.SpatialPoissonPointProcess` method to generate a point
    process. It is differ from parent class by having an :attr:`.expected` method (see: there).
    """

    def get(self, value, kernel, potential):
        """See: :meth:`.MarkovChainMonteCarloCompositeFunction.get`."""

        raise NotImplementedError

    @property
    def expected(self) -> float:
        """It should be the expected value of the implemented arithmetics.

        This allows :class:`.SpatialPoissonPointProcessIntensityFunction` to scale
        :attr:`~.SpatialPoissonPointProcessIntensityFunction.rate` so that the generated point
        process remain *similar* to a Poison point process.

        :return: It is the expected value of the composite function.
        :rtype: :class:`float`
        """

        raise NotImplementedError


# Implementation


class AdaptivePotentialFunction(MarkovChainMonteCarloPotentialFunction):
    """Implements a potential function that is a super-Gaussian distribution."""

    _monopole_potential_cache = {}

    def __init__(self, size: int, city_sizes: int) -> None:
        """Initializes a potential function.

        :param size: It is the size of a square bounding window.
        :type size: :class:`int`
        :param city_sizes: It is the number of levels cities can take.
        :type city_sizes: :class:`int`
        """

        self._city_sizes = city_sizes

        self._potential_image = Image.new('L', (size, size), 0)

    @classmethod
    def clear_cache(cls) -> None:
        """Clears the cached *monopole* values.

        See: :meth:`update`
        """

        cls._monopole_potential_cache.clear()

    @property
    def potential_map(self) -> Image.Image:
        """It is the current values of potential function over the bounding window."""

        return self._potential_image

    def get(self, value):
        """It is gives the value of the potential function.

        :param value: It is the point where the function is computed.
        :type value: :attr`.PointType`
        :return: It is the calculated value.
        :rtype: :class:`float`
        """

        x, y, _z = value
        pixel = (255 - self._potential_image.getpixel((x, y))) / 255
        return pixel

    def update(self, value):
        """It updates the function based on whether the candidate is accepted or rejected.

        Also updates the monopole cache if the monopole has not been previously calculated.
        A monopole is a collection of values the function is updated by superposing them over
        existing values. A monopole *m* is of a super-Gaussian distribution thus:

            *m* (*x0*, *y0*, *z*) ~ **e** ** (-((*x0* ** 2 + *y0* ** 2) / (2 * *z* ** 2)) ** *p*),
            where *p* = 1 / 3 and `value` = (*x*, *y*, *z*).

        :param value: It is the candidate where the potential function being updated.
        :type value: :attr:`.PointType`
        """

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
    """It is the implementation of an exponential function"""

    def get(self, value, kernel, potential):
        """It computes the product the following arithmetics:

            `potential` * `kernel` ** *z*,
            where `value` is (*x*, *y*, *z*).

        See also: :meth:`.MarkovChainMonteCarloCompositeFunction.get()`
        """
        rate = (potential * kernel) ** (value[2] + 1)
        return rate


class MarkovChainMonteCarloIntensityFunction(IntensityFunction):
    """It is an implementation of :class:`.IntensityFunction` used by
    :class:`~.point_process.MarkovChainMonteCarlo`.

    It is used for deciding thinning actions based on a
    :class:`MarkovChainMonteCarloCompositeFunction`.
    """

    def __init__(self, rate: Union[int, float], kernel_image: Image.Image,
                 potential_func: MarkovChainMonteCarloPotentialFunction,
                 composite_func: MarkovChainMonteCarloCompositeFunction) -> None:
        """Initializes an intensity function.

        :param rate: See: :class:`~.point_process.PointProcess`.
        :type rate: :data:`~typing.Union` [:class:`int`, :class:`float` ]
        :param kernel_image: It is an image containing the values of kernel function corresponding
            to pixel values in the **x** and **y** coordinates of the image.
        :type kernel_image: :class:`PIL.Image.Image`
        :param potential_func: It describes the pair-wise interaction of generated points.
        :type potential_func: :class:`.MarkovChainMonteCarloPotentialFunction`
        :param kernel_image: It contains the values of a kernel function.
        :type kernel_image: :class:`PIL.Image.Image`
        :param composite_func: It computes the final value of thinning process based of potential
            and kernel functions.
        :type composite_func: :class:`.MarkovChainMonteCarloCompositeFunction`
        """
        super().__init__()
        self._rate = rate
        self._potential_function = potential_func
        self._composite_function = composite_func
        self._kernel_image = kernel_image

    @property
    def rate(self):
        """It is the expected number of points generated.

        :return: It is the value set by `rate` during initialization.
        :rtype: :class:`float`
        """

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
    """It is an implementation of :class:`.IntensityFunction` used by
    :class:`~.point_process.SpatialPoissonPointProcess`.

    See: :class:`.MarkovChainMonteCarloIntensityFunction`.
    """

    def __init__(self, rate: Union[int, float], kernel_image: Image.Image,
                 potential_func: SpatialPoissonPointProcessPotentialFunction,
                 composite_func: SpatialPoissonPointProcessCompositeFunction) -> None:
        """Initializes an intensity function.

        Calculates mean of `kernel_image` to be used by
        :class:`SpatialPoissonPointProcessPotentialFunction`.

        See also: :class:`MarkovChainMonteCarloIntensityFunction`.

        :param potential_func:  It describes the pair-wise interaction of generated points.
        :type potential_func: :class:`.SpatialPoissonPointProcessPotentialFunction`
        :param composite_func:  It computes the final value of thinning process based of potential
            and kernel functions.
        :type composite_func: :class:`.SpatialPoissonPointProcessPotentialFunction`
        """
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
        return self._rate / self._composite_function.expected
