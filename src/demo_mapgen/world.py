import bisect
from dataclasses import dataclass, field, fields
from importlib import resources

from PIL import Image, ImageDraw, ImageStat

from . import data
from .common import safe_seed, SeedType
from .height_map import HeightMap
from .intensity import AdaptivePotentialFunction, ExponentialZCompositeFunction, \
    MarkovChainMonteCarloIntensityFunction
from .point_process import MarkovChainMonteCarlo


@dataclass(frozen=True)
class WorldConfig:
    chunk_size: int
    height: float
    roughness: float
    city_rate: int
    city_sizes: int = 1
    bit_length: int = 64

    # TODO make potential function, composite function and simulation method (intensity function) configurable

    def sanity_check(self):
        HeightMap.check(self.chunk_size, self.height, self.roughness)
        WorldChunk.check(self.city_rate, self.city_sizes)

        # early validation
        if self.chunk_size & self.chunk_size - 1 != 0:
            raise ValueError("Argument 'chunk_size' should be power of two")

        if self.bit_length & 1:
            raise ValueError("Argument 'bit_length' should be even")


@dataclass(frozen=True)
class WorldRenderOptions:
    show_debug: bool
    show_height_map: bool
    show_cities: bool
    show_potential_map: bool


default_world_config = WorldConfig(chunk_size=256, height=1., roughness=.5, city_rate=32, city_sizes=8)
default_render_options = WorldRenderOptions(False, True, True, False)


@dataclass(order=True, frozen=True)
class WorldChunkData:
    x: int
    y: int
    height_map: Image = field(compare=False)
    cities: list = field(compare=False)
    potential_map: Image = field(compare=False)


class WorldGenerator:

    def __init__(self, config: WorldConfig = default_world_config, seed: SeedType = None) -> None:
        config.sanity_check()
        self.config = config
        self._chunks = []

        self._seed = seed
        self._safe_seed = safe_seed(seed, self.config.bit_length)

    @property
    def seed(self):
        return self._seed if self._seed is not None else self._safe_seed

    def add_chunk(self, chunk_x: int, chunk_y: int):
        chunk = WorldChunk(chunk_x, chunk_y, self.config.chunk_size, self.config.height, self.config.roughness,
                           self.config.city_rate, self.config.city_sizes,
                           seed=self._safe_seed, bit_length=self.config.bit_length)
        chunk_data = chunk.generate()
        bisect.insort_right(self._chunks, chunk_data)

    @staticmethod
    def clear_potential_cache():
        AdaptivePotentialFunction.clear_cache()

    def render(self, *, options: WorldRenderOptions = default_render_options):
        if not self._chunks:
            raise IndexError("There are no chunks added to render")
        if not any(getattr(options, f.name) for f in fields(options)):
            raise ValueError("Argument 'option' sets nothing to render")

        x_max = max(self._chunks, key=lambda item: item.x).x
        x_min = min(self._chunks, key=lambda item: item.x).x
        y_max = max(self._chunks, key=lambda item: item.y).y
        y_min = min(self._chunks, key=lambda item: item.y).y

        width = (x_max - x_min + 1) * self.config.chunk_size
        height = (y_max - y_min + 1) * self.config.chunk_size
        r = 2
        colour = (255, 0, 0)
        border = (0, 0, 0)

        atlas_im = Image.new('RGBA', (width, height))
        draw_im = Image.new('RGBA', (width, height))
        potential_im = Image.new('L', (width, height))

        draw = ImageDraw.Draw(draw_im)
        palette = resources.open_binary(data.__name__, 'colourmap.palette').read()
        for chunk in self._chunks:
            cx = (chunk.x - x_min) * self.config.chunk_size
            cy = (chunk.y - y_min) * self.config.chunk_size

            if options.show_height_map:
                # concatenate heightmaps
                im = chunk.height_map.convert('P')
                im.putpalette(palette)
                atlas_im.paste(im, (cx, cy))
            if options.show_potential_map:
                potential_im.paste(chunk.potential_map, (cx, cy))

            if options.show_cities:
                # place cities
                for x, y, z in chunk.cities:
                    draw.ellipse(((cx + x - r - z, cy + y - r - z),
                                  (cx + x + r + z, cy + y + r + z)),
                                 fill=colour, outline=border, width=1)

            if options.show_debug:
                msg = '\n'.join((
                    f"count: {len(chunk.cities)}",
                    f"expected: {self.config.city_rate}",
                    f"mean: {(255 - ImageStat.Stat(chunk.height_map).mean.pop()) / 255:.3f}",
                    f"sizes: {self.config.city_sizes}")
                )
                draw.multiline_text((cx, cy), msg, fill=(0, 0, 0))
            atlas_im.paste(draw_im, mask=draw_im)

        if options.show_potential_map:
            alpha = Image.new('RGBA', (width, height), 0)
            alpha.putalpha(potential_im)
            atlas_im.alpha_composite(alpha)

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

        self._seed = safe_seed(seed, bit_length)
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

    def generate(self):
        height_map_image = self.height_map.generate()

        potential_function = AdaptivePotentialFunction(self._size, self.city_sizes)
        intensity_function = MarkovChainMonteCarloIntensityFunction(
            self.city_rate, height_map_image, potential_function, ExponentialZCompositeFunction()
        )
        volume = (self._size, self._size, self.city_sizes + 1)  # exclusive high
        mcmc = MarkovChainMonteCarlo(intensity_function, volume, self._local_seed)
        cities = [point for point in mcmc]

        # TODO: implement pathfinding

        world_data = WorldChunkData(
            self._chunk_x, self._chunk_y, height_map_image, cities, potential_function.potential_map
        )
        return world_data