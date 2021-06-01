"""Commonly used classes and functions"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Union

from PIL import Image

PointType = tuple[int, int, int]
"""Type of points (or cities)."""
SeedType = Union[None, int, str, bytes, bytearray]
"""Type of various seeds."""

# FIXME: remove noinspection comments


def get_safe_seed(seed: Any, bit_length: int) -> int:
    """Creates a safe integer seed

    :param seed: Only  `None`, `int`, `str`, `bytes`, and `bytearray` are supported types.
    :type seed: Any (technically)
    :param bit_length: Needs to be positive and less than 64.
    :type bit_length: int
    :return: A number safe to be used with :mod:`numpy.random` functions.
    :raises: `TypeError`, `ValueError`
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
    # noinspection PyUnresolvedReferences
    """Boilerplate for :class:`WorldGenerator`

    :param chunk_size: Needs to be power of 2 (see also: :class:`HeightMapConfig`).
    :type chunk_size: int
    :param height: (see: :class:`HeightMapConfig`)
    :type height: float
    :param roughness: (see: :class:`HeightMapConfig`)
    :type roughness: float
    :param city_rate: Needs to be positive (see: :class:`WorldChunk`).
    :type city_rate: int
    :param city_sizes: Needs to be positive, defaults to 1, (see: :class:`WorldChunk`).
    :type city_sizes: int
    :param bit_length: Needs to be positive, even, defaults to 64, (see: :class:`WorldChunk`).
    :type bit_length: int
    """

    chunk_size: int
    height: float
    roughness: float
    city_rate: int
    city_sizes: int = 1
    bit_length: int = 64

    def check(self) -> None:
        """Sanity check

        also checks :class:`HeightMapConfig`

        :raises: `TypeError`, `ValueError`
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
    # noinspection PyUnresolvedReferences
    """Boilerplate for :class:`HeightMap`

    :param size: Needs to be positive positive (corresponds to :attr:`WorldConfig.chunk_size`),
        (see also: :class:`WorldChunk`).
    :type size: int
    :param height: It's a probability, hence needs to be in [0, 1] (inclusive) range,
        (see also: :class:`HeightMap`).
    :type height: float
    :param roughness: It's a probability, hence needs to be in [0, 1] (inclusive) range,
        (see also: :class:`HeightMap`).
    :type roughness: float
    """

    size: int
    height: float
    roughness: float

    def check(self) -> None:
        """Sanity check

        :raises: `TypeError`, `ValueError`
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
    """Boilerplate for :meth:`WorldGenerator.render`"""

    show_debug: bool = False
    show_height_map: bool = True
    colour_height_map: bool = True
    show_cities: bool = True
    show_roads: bool = True
    show_potential_map: bool = False


@dataclass(order=True, frozen=True)
class WorldChunkData:
    # noinspection PyUnresolvedReferences
    """Boilerplate for :class:`WorldChunk` used by :class:`WorldGenerator`

    :param offset_x: *x* coordinate of chunk
    :type offset_x: int
    :param offset_y: *y* coordinate of chunk
    :type offset_y: int
    :param height_map: height map generated by :meth:`HeightMap.generate`
    :type height_map: :class:`Image.Image`
    :param cities: list of city coordinates generated by :class:`point_process.PointProcess`
    :type cities: list[:const:`PointType`, ...]
    :param potential_map: potential map generated by :meth:`intensity.AdaptivePotentialFunction`
    :type potential_map: :class:`Image.Image`
    :param pixel_paths: list of coordinates generated by :func:`pathfinder.find_shortest_paths`
    :type pixel_paths: dict[tuple[:const:`PointType`, :const:`PointType`], :class:`PixelPath`]
    """

    offset_x: int
    offset_y: int
    height_map: Image.Image = field(compare=False)
    cities: list[PointType, ...] = field(compare=False)
    potential_map: Image.Image = field(compare=False)
    pixel_paths: dict[tuple[PointType, PointType], PixelPath] = field(compare=False)


@dataclass(frozen=True)
class WorldData:
    # noinspection PyUnresolvedReferences
    """Boilerplate for :class:`WorldGenerator` used by :class:`Datafile`

    :param config: Value of :attr:`WorldGenerator.config`.
    :type config: :class:`WorldConfig`
    :param seed: Value of :attr:`WorldGenerator._seed`.
    :type seed: :const:`SeedType`
    :param safe_seed: :attr:`WorldGenerator._safe_seed`.
    :type safe_seed: int
    :param chunks: Value of :attr:`WorldGenerator._chunks`.
    :type chunks: list[:class:`WorldChunkData`, ...]
    """

    config: WorldConfig
    seed: SeedType
    safe_seed: int
    chunks: list[WorldChunkData, ...]


@dataclass(frozen=True)
class PixelPath:
    # noinspection PyUnresolvedReferences
    """Boilerplate for :meth:`pathfinder.Pathfinder.shortest_paths`

    :param cost: Total cost of the path, including diagonal and vertical cost.
    :type cost: float
    :param pixels: A list of each pixel the path contains.
    :type pixels: list[tuple[int, int]]
    """

    cost: float
    pixels: list[tuple[int, int]]
