from __future__ import annotations

import hashlib
import hmac
import pickle
import struct
from dataclasses import dataclass, field
from typing import Union

from PIL import Image, ImageDraw, ImageStat

from . import __version_info__ as package_version
from .common import get_safe_seed, SeedType, PointType
from .data import colour_palette
from .height_map import HeightMap
from .intensity import AdaptivePotentialFunction, ExponentialZCompositeFunction, \
    MarkovChainMonteCarloIntensityFunction
from .pathfinder import find_shortest_paths, PixelPath
from .point_process import MarkovChainMonteCarlo


@dataclass(frozen=True)
class WorldConfig:
    chunk_size: int
    height: float
    roughness: float
    city_rate: int
    city_sizes: int = 1
    bit_length: int = 64

    def check(self):
        HeightMap.check(self.chunk_size, self.height, self.roughness)
        WorldChunk.check(self.city_rate, self.city_sizes)

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
    pixel_paths: list[PixelPath, ...] = field(compare=False)


default_world_config = WorldConfig(chunk_size=256, height=1., roughness=.5, city_rate=32, city_sizes=8)
default_render_options = WorldRenderOptions()


class WorldGeneratorDatafile:
    # TODO: test file format:
    # [x] * 10 bytes * magic number
    # [x] *  3 bytes * version number
    # [x] * 64 bytes * header checksum
    # [x] *  2 bytes * data offset
    # --- *  1 byte  * pad ---
    # header:
    # [x] *  4 bytes * content length (32Gb data max)
    # [x] * 64 bytes * signature (see bellow)
    # [ ] other things, maybe
    # --- *  1 byte  * pad ---
    # data:
    # [x] *  1 bit   * is seed a string
    # [x] *  1 byte  * length of seed
    # [x] *  varied  * self._seed
    # --- *  1 byte  * pad ---
    # [x] *  8 bytes * self._safe_seed
    # --- *  1 byte  * pad ---
    # [x] *  1 byte  * length of config (future proof, currently 43)
    # [x] *  varied  * config
    # --- *  1 byte  * separator ---
    # chunks:
    # [x] *  2 bytes * length of chunks
    # [?] *  1 byte  * pad ---
    # chunk:
    # [x] *  4 bytes * x
    # [x] *  4 bytes * y
    # [!] *  2 bytes * length of cities (is it enough?)
    # [?] *  varied  * cities
    # [?] *  1 byte  * pad ---
    # [?] *  2 bytes * length of pixel_paths
    # [?] *  1 byte  * separator ---
    # pixel_path:
    # [!] *  4 bytes * cost (is it correct?)
    # [!] *  2 bytes * length of pixels (is it enough?)
    # [?] *  varied  * pixels
    # [?] *  1 byte  * separator ---
    # [?] *  2 bytes * length of height_map
    # [?] *  varied  * height_map
    # [?] *  1 byte  * separator ---
    # [?] *  2 bytes * length of potential_map
    # [?] *  varied  * potential_map
    # [x] *  1 byte * separator --- (if not last)
    # [ ] EOF (might not be needed, note signature)

    __magic = '#D-MG#WG-D'.encode('ascii')

    def __init__(self):
        self._data: bytes = bytes()

    # TODO: implement method for data to be validated

    def add_data(self, seed: SeedType, safe_seed: int, config: WorldConfig, chunks: list[WorldChunkData, ...]):
        if seed is None:
            is_seed_str = False
            seed = b''
        elif isinstance(seed, int):
            is_seed_str = False
            seed = seed.to_bytes((seed.bit_length() + 7) // 8, byteorder='little')
        elif isinstance(seed, str):
            is_seed_str = True
            seed = seed.encode()
        else:  # bytes or bytearray
            is_seed_str = False
        seed = seed[:255]  # first 255 bytes (if larger)
        config = pickle.dumps(list(config.__dict__.values()))

        self._data = b'\00'.join((
            struct.pack(
                '<?B%dsxQxB%dsH' % (len(seed), len(config)),
                is_seed_str,  # 1 bit
                len(seed),  # 1 byte
                seed,  # varied
                # pad 1 byte
                safe_seed,  # 8 bytes
                # pad 1 byte
                len(config),  # 1 byte
                config,  # 43 bytes
                len(chunks)  # 2 bytes
            ),
            # separator 1 byte
            b'\00'.join(self._encode_chunk_data(chunk) for chunk in chunks)
        ))

    def save(self, filename: Union[str, bytes], key: bytes):
        with open(filename, 'wb') as f:
            content_length = len(self._data)
            signature = hmac.new(key, self._data, hashlib.sha512).digest()
            header = struct.pack(
                '<xL%dsx' % len(signature),  # expected 64 (`hmac` `digest_size` with `sha512` method)
                # pad 1 bytes
                content_length,  # 4 bytes
                signature,  # 64 bytes
                # pad 1 byte
            )
            checksum = hashlib.sha512(header).digest()
            magic = struct.pack(
                '<10s3B%dsH' % len(checksum),  # expected 64 (`hashlib` `sha512` method `digest_size`)
                self.__magic,  # 10 bytes
                *package_version,  # 3 bytes
                checksum,  # 64 bytes
                len(header),  # 2 bytes
                # pad 1 byte
            )
            f.write(magic)
            f.write(header)
            f.write(self._data)

    @staticmethod
    def _encode_chunk_data(chunk: WorldChunkData):
        cities = pickle.dumps(chunk.cities)
        height_map = chunk.height_map.tobytes()
        potential_map = chunk.potential_map.tobytes()

        data = b'\00'.join((
            struct.pack(
                '<iiH%dsxH' % len(cities),
                chunk.x,  # 4 bytes
                chunk.y,  # 4 bytes
                len(cities),  # 2 bytes
                cities,
                # pad 1 byte
                len(chunk.pixel_paths)  # 2 bytes
            ),
            # separator 1 byte
            b'\00'.join(WorldGeneratorDatafile._encode_pixel_path(path) for path in chunk.pixel_paths),
            # separator 1 byte
            struct.pack(
                '<H%ds',
                len(height_map),
                height_map
            ),
            # separator 1 byte
            struct.pack(
                '<H%ds',
                len(potential_map),
                potential_map
            )
        ))
        return data

    @staticmethod
    def _encode_pixel_path(path: PixelPath):
        pixels = pickle.dumps(path.pixels)
        data = struct.pack(
            '<iH%ds',
            path.cost,  # 4 bytes
            len(pixels),  # 2 bytes
            pixels
        )
        return data


class WorldGenerator:

    def __init__(self, *, config: WorldConfig = default_world_config, seed: SeedType = None) -> None:
        config.check()
        self.config = config
        self._chunks: list[WorldChunkData, ...] = []

        self._seed = seed if isinstance(seed, int) or seed is None else seed[:255]  # for i/o compatibility
        self._safe_seed = get_safe_seed(seed, self.config.bit_length)

    @property
    def seed(self) -> SeedType:
        return self._seed if self._seed is not None else self._safe_seed

    @staticmethod
    def clear_potential_cache():
        AdaptivePotentialFunction.clear_cache()

    # TODO: implement i/o methods
    # using following file format:
    # - magic number followed by data offset,
    # - header containing: version number and signature (see bellow) ... other things maybe
    # - data to be serialized:
    #      self._seed, self._safe_seed
    #      config (including: used functions from `intensity`, possibly names only)
    #      chunks (including: cities, cost and pixels of pixel_paths, height_map and potential_map images)
    # - data needs to be signed (validated)

    def add_chunk(self, chunk_x: int, chunk_y: int):
        chunk = WorldChunk(chunk_x, chunk_y, self.config.chunk_size, self.config.height, self.config.roughness,
                           self.config.city_rate, self.config.city_sizes,
                           seed=self._safe_seed, bit_length=self.config.bit_length)
        chunk_data = chunk.generate()
        self._chunks.append(chunk_data)

    def render(self, *, options: WorldRenderOptions = default_render_options):
        if not self._chunks:
            raise IndexError("There are no chunks added to render")
        if not any((options.show_debug, options.show_height_map, options.show_cities, options.show_roads,
                    options.show_potential_map)):
            raise ValueError("Nothing to render with given 'option' argument")

        x_max = max(self._chunks, key=lambda item: item.x).x
        x_min = min(self._chunks, key=lambda item: item.x).x
        y_max = max(self._chunks, key=lambda item: item.y).y
        y_min = min(self._chunks, key=lambda item: item.y).y

        width = (x_max - x_min + 1) * self.config.chunk_size
        height = (y_max - y_min + 1) * self.config.chunk_size

        city_r = 2
        city_colour = (255, 0, 0)
        city_border = (0, 0, 0)
        text_color = (0, 0, 0)

        atlas_im = Image.new('RGBA', (width, height))
        draw_im = Image.new('RGBA', (width, height))

        draw = ImageDraw.Draw(draw_im)

        for chunk in self._chunks:
            cx = (chunk.x - x_min) * self.config.chunk_size
            cy = (chunk.y - y_min) * self.config.chunk_size

            # concatenate height maps
            if options.show_height_map:
                if options.colour_height_map:
                    im = chunk.height_map.convert('P')
                    im.putpalette(colour_palette)
                else:
                    im = chunk.height_map
                atlas_im.paste(im, (cx, cy))

            # overlay potential field
            if options.show_potential_map:
                alpha = Image.new('RGBA', (self.config.chunk_size, self.config.chunk_size), 0)
                alpha.putalpha(chunk.potential_map)
                atlas_im.alpha_composite(alpha, (cx, cy))

            # place cities
            if options.show_cities:
                for x, y, z in chunk.cities:
                    draw.ellipse((cx + x - city_r - z, cy + y - city_r - z,
                                  cx + x + city_r + z, cy + y + city_r + z),
                                 fill=city_colour, outline=city_border, width=1)

            # put message in top left corner
            if options.show_debug:
                msg = '\n'.join((
                    f"count: {len(chunk.cities)}",
                    f"expected: {self.config.city_rate}",
                    f"mean: {(255 - ImageStat.Stat(chunk.height_map).mean.pop()) / 255:.3f}",
                    f"sizes: {self.config.city_sizes}")
                )
                draw.multiline_text((cx, cy), msg, fill=text_color)

            # draw roads
            if options.show_roads:
                # XXX: avoiding `Image.Image.putpixel`
                path_data = [0] * (self.config.chunk_size * self.config.chunk_size)
                for path in chunk.pixel_paths:
                    for point_x, point_y in path.pixels:
                        path_data[point_x + point_y * self.config.chunk_size] = 255
                im = Image.new('RGBA', (self.config.chunk_size, self.config.chunk_size), 0)
                im.putalpha(Image.frombytes('L', (self.config.chunk_size, self.config.chunk_size), bytes(path_data)))
                draw_im.paste(im, (cx, cy), mask=im)

        atlas_im.paste(draw_im, mask=draw_im)
        return atlas_im


class WorldChunk:

    def __init__(self, chunk_x: int, chunk_y: int, size: int, height: float, roughness: float,
                 city_rate: int, city_sizes: int, *,
                 seed: SeedType = None, bit_length: int = 64) -> None:
        self.check(city_rate, city_sizes)

        self._chunk_x = chunk_x
        self._chunk_y = chunk_y

        self._height_map = HeightMap(chunk_x, chunk_y, size, height, roughness, seed=seed, bit_length=bit_length)
        self._size = self._height_map.size

        self.city_sizes = city_sizes
        self.city_rate = city_rate

        self._seed = get_safe_seed(seed, bit_length)
        x = self._chunk_x * self._size
        y = self._chunk_y * self._size
        seed = (x ^ y << (bit_length >> 1)) ^ self._seed
        self._local_seed = seed & ((1 << bit_length) - 1)

    @staticmethod
    def check(city_rate, city_sizes):
        if not isinstance(city_rate, int):
            raise TypeError("Argument 'city_rate' should be integer number, not '%s'" % type(city_rate).__name__)
        if city_rate <= 0:
            raise ValueError("Argument 'city_rate' should be positive")
        if not isinstance(city_sizes, int):
            raise TypeError("Argument 'city_sizes' should be integer number, not '%s'" % type(city_sizes).__name__)
        if city_sizes <= 0:
            raise ValueError("Argument 'city_sizes' should be positive")

    @property
    def size(self) -> int:
        return self._size

    @property
    def height_map(self) -> HeightMap:
        return self._height_map

    def generate(self) -> WorldChunkData:
        height_map_image = self.height_map.generate()

        potential_function = AdaptivePotentialFunction(self._size, self.city_sizes)
        intensity_function = MarkovChainMonteCarloIntensityFunction(
            self.city_rate, height_map_image, potential_function, ExponentialZCompositeFunction()
        )
        volume = (self._size, self._size, self.city_sizes + 1)  # exclusive high
        mcmc = MarkovChainMonteCarlo(intensity_function, volume, self._local_seed)
        cities = [point for point in mcmc]

        paths = {}
        for i, source in enumerate(cities[:-1], 1):
            paths.update(find_shortest_paths(height_map_image, source, cities[i:]))

        selected_path = [max(paths.values(), key=lambda path: path.cost)]

        world_data = WorldChunkData(
            self._chunk_x, self._chunk_y, height_map_image, cities, potential_function.potential_map, selected_path
        )
        return world_data
