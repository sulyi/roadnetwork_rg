from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Union

from PIL import Image

PointType = tuple[int, int, int]
SeedType = Union[None, int, str, bytes, bytearray]


def world_generator_check(city_rate: int, city_sizes: int) -> None:
    if not isinstance(city_rate, int):
        raise TypeError("Argument 'city_rate' should be integer number, not '%s'" %
                        type(city_rate).__name__)
    if city_rate <= 0:
        raise ValueError("Argument 'city_rate' should be positive")

    if not isinstance(city_sizes, int):
        raise TypeError("Argument 'city_sizes' should be integer number, not '%s'" %
                        type(city_sizes).__name__)
    if city_sizes <= 0:
        raise ValueError("Argument 'city_sizes' should be positive")


def height_map_check(size: int, height: float, roughness: float) -> None:
    if not isinstance(size, int):
        raise TypeError("Argument 'size' should be integer number, not '%s'" %
                        type(size).__name__)
    if size <= 0:
        raise ValueError("Argument 'size' should be positive")

    if not isinstance(height, (float, int)):
        raise TypeError("Argument 'height' should be a number, not '%s'" %
                        type(height).__name__)
    if not 0 <= height <= 1:
        raise ValueError("Argument 'height' should be in [0, 1] inclusive range")

    if not isinstance(roughness, (float, int)):
        raise TypeError("Argument 'roughness' should be a number, not '%s'" %
                        type(roughness).__name__)
    if not 0 <= roughness <= 1:
        raise ValueError("Argument 'roughness' should be in [0, 1] inclusive range")


def get_safe_seed(seed: Any, bit_length: int) -> int:
    if not isinstance(bit_length, int):
        raise TypeError("Argument 'bit_length' should be integer number, not '%s'" %
                        type(bit_length).__name__)
    if bit_length <= 0:
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
    chunk_size: int
    height: float
    roughness: float
    city_rate: int
    city_sizes: int = 1
    bit_length: int = 64

    def check(self) -> None:
        height_map_check(self.chunk_size, self.height, self.roughness)
        world_generator_check(self.city_rate, self.city_sizes)

        # early validation
        if self.chunk_size & self.chunk_size - 1 != 0:
            raise ValueError("Argument 'chunk_size' should be power of two")

        if self.bit_length & 1:
            raise ValueError("Argument 'bit_length' should be even")


@dataclass(frozen=True)
class WorldRenderOptions:
    show_debug: bool = False
    show_height_map: bool = True
    colour_height_map: bool = True
    show_cities: bool = True
    show_roads: bool = True
    show_potential_map: bool = False


@dataclass(order=True, frozen=True)
class WorldChunkData:
    x: int
    y: int
    height_map: Image.Image = field(compare=False)
    cities: list[PointType, ...] = field(compare=False)
    potential_map: Image.Image = field(compare=False)
    pixel_paths: dict[tuple[PointType, PointType], PixelPath] = field(compare=False)


@dataclass(frozen=True)
class WorldData:
    config: WorldConfig
    seed: SeedType
    safe_seed: int
    chunks: list[WorldChunkData, ...]


@dataclass(frozen=True)
class PixelPath:
    cost: float
    pixels: list[tuple[int, int]]
