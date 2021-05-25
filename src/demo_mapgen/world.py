from __future__ import annotations

from dataclasses import dataclass, field

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
    # TODO: implement file format:
    # [x] * 10 bytes * magic number
    # [x] *  2 bytes * data offset
    # [ ] header checksum
    # --- header ---
    # [x] *  3 bytes * version number
    # [x] *  4 bytes * content length
    # [ ] signature (see bellow)
    # [ ] ... other things, maybe
    # --- data ---
    # [ ] self._seed
    # [ ] self._safe_seed
    # [ ] config (including: used functions from `intensity`, possibly names only)
    # [ ] chunks (including: cities, cost and pixels of pixel_paths, height_map and potential_map images)

    __magic = '#D-MG#WG-D'.encode('ascii')

    def __init__(self):
        self._data = None

    # TODO: implement method for data to be signed and validated

    def add_data(self, seed: SeedType, safe_seed: int, config: WorldConfig, chunks: list[WorldChunkData, ...]):
        pass

    def save(self, filename):
        with open(filename, 'wb') as f:
            data = bytes() if self._data is None else self._data
            content_length = 0
            header = b''.join((
                b''.join(x.to_bytes(1, 'little') for x in package_version),  # 3 byte
                content_length.to_bytes(4, 'little')  # 4 byte
            ))
            f.write(b''.join((
                self.__magic,  # 10 byte
                len(header).to_bytes(2, 'little'),  # 2 byte
                header,
                data
            )))


class WorldGenerator:

    def __init__(self, *, config: WorldConfig = default_world_config, seed: SeedType = None) -> None:
        config.check()
        self.config = config
        self._chunks: list[WorldChunkData, ...] = []

        self._seed = seed
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
