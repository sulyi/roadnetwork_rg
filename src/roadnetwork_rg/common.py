"""Commonly used classes and functions"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union, Tuple

from PIL import Image

PointType = Tuple[int, int, int]
"""It is the type of points (or cities)."""
SeedType = Union[None, int, str, bytes, bytearray]
"""It is the type of various seeds."""


def get_safe_seed(seed: Any, bit_length: int) -> int:
    """Creates a safe integer seed.

    :param seed: Only  :class:`None`, :class:`int`, :class:`str`, :class:`bytes`, and
        :class:`bytearray` are supported types.
    :type seed: :class:`Any` (technically)
    :param bit_length: Needs to be positive and less than 64.
    :type bit_length: :class:`int`
    :return: A number safe to be used with :mod:`numpy.random` functions.
    :raises: :exc:`TypeError`, :exc:`ValueError`
    """
    if not isinstance(bit_length, int):
        raise TypeError("Argument 'bit_length' should be integer number, not '%s'" %
                        type(bit_length).__name__)
    if bit_length < 0:
        raise ValueError("Argument 'bit_length' should be positive")
    if bit_length > 64:
        raise ValueError("Argument 'bit_length' should be less than 64")

    if seed is None:
        seed = random.randint(0, 1 << bit_length - 1)  # signed
    elif not isinstance(seed, int):
        if isinstance(seed, (str, bytes, bytearray)):
            if isinstance(seed, str):
                seed = seed.encode()
            seed += hashlib.sha512(seed).digest()
            seed = int.from_bytes(seed, 'big')
        else:
            raise TypeError("The only supported seed types are: "
                            "None, int, str, bytes, and bytearray")
    return seed & (1 << bit_length) - 1  # masked


@dataclass(frozen=True)
class WorldConfig:
    """It is a boilerplate used by :class:`.WorldGenerator`.

    :param chunk_size: Needs to be power of 2 (see also: :class:`HeightMapConfig`).
    :type chunk_size: :class:`int`
    :param height: (see: :class:`HeightMapConfig`)
    :type height: :class:`float`
    :param roughness: (see: :class:`HeightMapConfig`)
    :type roughness: :class:`float`
    :param city_rate: It is the expected number of cities, needs to be positive.
    :type city_rate: :class:`int`
    :param city_sizes: It is the number of sizes cities can take, needs to be positive,
        defaults to 1.
    :type city_sizes: :class:`int`
    :param bit_length: Needs to be positive and even, defaults to 64
        (see: :class:`.HeightMap`).
    :type bit_length: :class:`int`
    """
    # FIXME: update doc, update check

    chunk_size: int
    height: float
    roughness: float
    city_rate: int
    intensity_function: str
    potential_function: str
    point_process: str
    filter: str
    city_sizes: int = 1
    bit_length: int = 64

    def check(self) -> None:
        """Performs sanity check.

        also checks :class:`HeightMapConfig`

        :raises: :exc:`TypeError`, :exc:`ValueError`
        """

        HeightMapConfig(self.chunk_size, self.height, self.roughness).check()

        if not isinstance(self.city_rate, int):
            raise TypeError("Argument 'city_rate' should be integer number, not '%s'" %
                            type(self.city_rate).__name__)
        if self.city_rate < 0:
            raise ValueError("Argument 'city_rate' should be positive")

        if not isinstance(self.city_sizes, int):
            raise TypeError("Argument 'city_sizes' should be integer number, not '%s'" %
                            type(self.city_sizes).__name__)
        if self.city_sizes < 0:
            raise ValueError("Argument 'city_sizes' should be positive")

        # early validation
        if self.bit_length < 0:
            raise ValueError("Argument 'bit_length' should be positive")
        if self.bit_length & 1:
            raise ValueError("Argument 'bit_length' should be even")

        if self.chunk_size & self.chunk_size - 1 != 0:
            raise ValueError("Argument 'chunk_size' should be power of two")


@dataclass(frozen=True)
class HeightMapConfig:
    """It is a boilerplate used by :class:`.HeightMap`.

    :param size: It is the size of height map, needs to be positive positive (corresponds to
        `chunk_size` attribute of :class:`WorldConfig`). Actual value used is the smallest power of
        two larger or equal to this.
    :type size: :class:`int`
    :param height: It is the amount of displacement in first step, needs to be in [0, 1] (inclusive)
        range.
    :type height: :class:`float`
    :param roughness: It is the ratio by witch amount of displacement is changed in each step, needs
        to be in [0, 1] inclusive) range.
    :type roughness: :class:`float`
    """

    size: int
    height: float
    roughness: float

    def check(self) -> None:
        """Performs sanity check

        :raises: :exc:`TypeError`, :exc:`ValueError`
        """

        if not isinstance(self.size, int):
            raise TypeError("Argument 'size' should be integer number, not '%s'" %
                            type(self.size).__name__)
        if self.size < 0:
            raise ValueError("Argument 'size' should be positive")

        if not isinstance(self.height, (float, int)):
            raise TypeError("Argument 'height' should be a number, not '%s'" %
                            type(self.height).__name__)
        if not 0 <= self.height <= 1:
            raise ValueError("Argument 'height' should be in [0, 1] inclusive range")

        if not isinstance(self.roughness, (float, int)):
            raise TypeError("Argument 'roughness' should be a number, not '%s'" %
                            type(self.roughness).__name__)
        if not 0 <= self.roughness <= 1:
            raise ValueError("Argument 'roughness' should be in [0, 1] inclusive range")


@dataclass(frozen=True)
class WorldRenderOptions:
    """It is a boilerplate used by :meth:`.WorldGenerator.render`.

    see also: :meth:`.WorldGenerator.render`

    :param show_debug: When it is :data:`True` information about each chunk is rendered in
        its upper-left corner e.g: number of cities, mean of height map ..., defaults to
        :data:`False`.
    :type show_debug: :class:`bool`
    :param show_height_map:  When it is :data:`True` height map is rendered, defaults to
        :data:`True`.
    :type  show_height_map: :class:`bool`
    :param colour_height_map: When it is :data:`True` the :data:`~.data.colour_palette` is
        applied to height map before rendering it, otherwise it is rendered as grayscale image,
        defaults to :data:`True`.
    :type  colour_height_map: :class:`bool`
    :param show_cities: When it is :data:`True` cities are rendered, defaults to :data:`True`.
    :type show_cities: :class:`bool`
    :param show_roads: When it is :data:`True` roads are rendered, defaults to :data:`True`.
    :type show_roads: :class:`bool`
    :param show_potential_map: When it is :data:`True` final image is overlaid with
        potential map, defaults to :data:`False`.
    :type show_potential_map: :class:`bool`
    """

    show_debug: bool = False
    show_height_map: bool = True
    colour_height_map: bool = True
    show_cities: bool = True
    show_roads: bool = True
    show_potential_map: bool = False


@dataclass(order=True, frozen=True)
class WorldChunkData:
    """It is a boilerplate used by :class:`.WorldChunk` used by :class:`.WorldGenerator`.

    :param offset_x: It is the *x* coordinate of the chunk.
    :type offset_x: :class:`int`
    :param offset_y: It is the *y* coordinate of the chunk.
    :type offset_y: :class:`int`
    :param height_map: It is a height map generated by :meth:`.HeightMap.generate`.
    :type height_map: :class:`PIL.Image.Image`
    :param cities: It is a list of coordinates generated by :class:`~.point_process.PointProcess`.
    :type cities: :class:`list` [:const:`PointType`, ...]
    :param potential_map: It is a potential map generated by
        :class:`~.intensity.AdaptivePotentialFunction`.
    :type potential_map: :class:`PIL.Image.Image`
    :param pixel_paths: It is a list of coordinates generated by
        :meth:`.Pathfinder.shortest_paths`
    :type pixel_paths: :class:`dict` [:class:`tuple` [:const:`PointType`, :const:`PointType`],
        :class:`PixelPath`]
    """

    offset_x: int
    offset_y: int
    height_map: Image.Image = field(compare=False)
    cities: List[PointType, ...] = field(compare=False)
    potential_map: Image.Image = field(compare=False)
    pixel_paths: Dict[Tuple[PointType, PointType], PixelPath] = field(compare=False)


@dataclass(frozen=True)
class WorldData:
    """It is a boilerplate used by :class:`.WorldGenerator` used by :class:`.Datafile`.

    :param config: Value of :attr:`config` of :class:`.WorldGenerator`.
    :type config: :class:`WorldConfig`
    :param seed: Value of :attr:`_seed` of :class:`.WorldGenerator`.
    :type seed: :const:`SeedType`
    :param safe_seed: Value of :attr:`_safe_seed` of :class:`.WorldGenerator`.
    :type safe_seed: :class:`int`
    :param chunks: Value of :attr:`_chunks` of :class:`.WorldGenerator`.
    :type chunks: :class:`list` [:class:`WorldChunkData`, ...]
    """

    config: WorldConfig
    seed: SeedType
    safe_seed: int
    chunks: List[WorldChunkData, ...]


@dataclass(frozen=True)
class PixelPath:
    """It is a boilerplate used by :meth:`~.pathfinder.Pathfinder.shortest_paths`.

    :param cost: Total cost of the path, including diagonal and vertical cost.
    :type cost: :class:`float`
    :param pixels: A list of each pixel the path contains.
    :type pixels: :class:`list` [:class:`tuple` [:class:`int`, :class:`int` ]]
    """

    cost: float
    pixels: List[Tuple[int, int]]
