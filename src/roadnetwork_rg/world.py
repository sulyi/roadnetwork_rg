from __future__ import annotations

import warnings
from typing import Union

from PIL import Image, ImageDraw, ImageStat

from .common import (HeightMapConfig, SeedType, WorldChunkData, WorldConfig, WorldData,
                     WorldRenderOptions, get_safe_seed)
from .data import colour_palette
from .datafile import Datafile
from .height_map import HeightMap
from .intensity import (AdaptivePotentialFunction, ExponentialZCompositeFunction,
                        MarkovChainMonteCarloIntensityFunction)
from .pathfinder import Pathfinder
from .point_process import MarkovChainMonteCarlo

default_world_config = WorldConfig(chunk_size=256, height=1., roughness=.5,
                                   city_rate=32, city_sizes=8)
default_render_options = WorldRenderOptions()


class WorldGenerator:
    __city_r = 2
    __city_colour = (255, 0, 0)
    __text_color = (0, 0, 0)
    __city_border = (0, 0, 0)

    def __init__(self, *, config: WorldConfig = default_world_config,
                 seed: SeedType = None) -> None:
        config.check()
        self._config = config
        self._chunks: dict[tuple[int, int], WorldChunkData, ...] = {}

        # for i/o compatibility truncate seed to 255
        self._seed = seed if isinstance(seed, int) or seed is None else seed[:255]
        self._safe_seed = get_safe_seed(seed, self._config.bit_length)

    @classmethod
    def save(cls, instance: WorldGenerator, filename: Union[str, bytes], key: bytes) -> None:
        if not isinstance(instance, cls):
            raise ValueError("Argument is not an %s object" % cls.__name__)
        instance.write(filename, key)

    @classmethod
    def load(cls, filename: Union[str, bytes], key: bytes) -> WorldGenerator:
        instance = cls.__new__(cls)
        instance.read(filename, key)
        return instance

    @property
    def config(self):
        return self._config

    @property
    def seed(self) -> SeedType:
        return self._seed if self._seed is not None else self._safe_seed

    def get_chunks(self) -> list[WorldChunkData, ...]:
        return list(self._chunks.values())

    def read(self, filename: Union[str, bytes], key: bytes) -> None:
        data = Datafile.load(filename, key).get_data()
        dirty = (data.seed is not None and
                 data.safe_seed != get_safe_seed(data.seed, data.config.bit_length))

        if dirty:
            seed = None
            warnings.warn("Mismatching seed and safe_seed, seed is discarded")
        else:
            seed = data.seed

        try:
            data.config.check()
        except (ValueError, TypeError) as err:
            config = default_world_config
            warnings.warn("Failed configuration check, due to:")
            warnings.warn("%s" % err, stacklevel=2)
            warnings.warn("config set to default config")
            dirty = True
        else:
            config = data.config

        self._config = config
        self._chunks = {(chunk.offset_x, chunk.offset_y): chunk for chunk in data.chunks}

        self._seed = seed
        self._safe_seed = data.safe_seed

        if dirty:
            # FIXME: add specific exception
            raise Exception("Some error occurred during loading file")

    def write(self, filename: Union[str, bytes], key: bytes) -> None:
        Datafile.save(filename, key,
                      WorldData(self._config, self._seed, self._safe_seed, self.get_chunks()))

    def add_chunk(self, chunk_x: int, chunk_y: int) -> None:
        if (chunk_x, chunk_y) in self._chunks:
            warnings.warn("Chunk already exists")
            return

        chunk = WorldChunk(chunk_x, chunk_y, self._config, seed=self._safe_seed,
                           bit_length=self._config.bit_length)
        chunk_data = chunk.generate()
        self._chunks[chunk_x, chunk_y] = chunk_data

    def render(self, *, options: WorldRenderOptions = default_render_options) -> Image.Image:
        if not self._chunks:
            raise IndexError("There are no chunks added to render")
        if not any((options.show_debug, options.show_height_map, options.show_cities,
                    options.show_roads, options.show_potential_map)):
            raise ValueError("Nothing to render with given 'option' argument")

        min_x = min(self._chunks.values(), key=lambda item: item.offset_x).offset_x
        min_y = min(self._chunks.values(), key=lambda item: item.offset_y).offset_y

        size = (
            (max(self._chunks.values(),
                 key=lambda item: item.offset_x).offset_x - min_x + 1) * self._config.chunk_size,
            (max(self._chunks.values(),
                 key=lambda item: item.offset_y).offset_y - min_y + 1) * self._config.chunk_size
        )
        atlas_im = Image.new('RGBA', size)
        draw_im = Image.new('RGBA', size)

        draw = ImageDraw.Draw(draw_im)

        for chunk in self._chunks.values():
            cx = (chunk.offset_x - min_x) * self._config.chunk_size
            cy = (chunk.offset_y - min_y) * self._config.chunk_size

            # concatenate height maps
            atlas_im.paste(self._render_height_map(chunk, options), (cx, cy))
            # overlay potential field
            atlas_im.alpha_composite(self._render_potential_map(chunk, options), (cx, cy))

            # place cities
            if options.show_cities:
                for x, y, z in chunk.cities:
                    draw.ellipse((cx + x - self.__city_r - z, cy + y - self.__city_r - z,
                                  cx + x + self.__city_r + z, cy + y + self.__city_r + z),
                                 fill=self.__city_colour, outline=self.__city_border, width=1)

            # put message in top left corner
            if options.show_debug:
                draw.multiline_text(
                    (cx, cy),
                    '\n'.join((
                        f"count: {len(chunk.cities)}",
                        f"expected: {self._config.city_rate}",
                        f"mean: {(255 - ImageStat.Stat(chunk.height_map).mean.pop()) / 255:.3f}",
                        f"sizes: {self._config.city_sizes}")
                    ),
                    fill=self.__text_color
                )

            # draw roads
            image = self._render_draw_roads(chunk, options)
            draw_im.paste(image, (cx, cy), mask=image)

        atlas_im.paste(draw_im, mask=draw_im)
        return atlas_im

    @staticmethod
    def _render_height_map(chunk: WorldChunkData, options: WorldRenderOptions) -> Image.Image:
        if options.show_height_map:
            if options.colour_height_map:
                image = chunk.height_map.convert('P')
                image.putpalette(colour_palette)
            else:
                image = chunk.height_map
        else:
            image = Image.new('RGBA', chunk.height_map.size)
        return image

    @staticmethod
    def _render_potential_map(chunk: WorldChunkData, options: WorldRenderOptions) -> Image.Image:
        if options.show_potential_map:
            alpha = Image.new('RGBA', chunk.potential_map.size, 0)
            alpha.putalpha(chunk.potential_map)
        else:
            alpha = Image.new('RGBA', chunk.height_map.size)
        return alpha

    @staticmethod
    def _render_draw_roads(chunk: WorldChunkData, options: WorldRenderOptions) -> Image.Image:
        if options.show_roads:
            # XXX: avoiding `Image.Image.putpixel`
            path_data = [0] * (chunk.height_map.size[0] * chunk.height_map.size[1])
            for path in chunk.pixel_paths.values():
                for point_x, point_y in path.pixels:
                    path_data[point_x + point_y * chunk.height_map.size[0]] = 255
            image = Image.new('RGBA', chunk.height_map.size, 0)
            image.putalpha(Image.frombytes('L', chunk.height_map.size,
                                           bytes(path_data)))
        else:
            image = Image.new('RGBA', chunk.height_map.size)
        return image

    @staticmethod
    def clear_potential_cache() -> None:
        AdaptivePotentialFunction.clear_cache()


class WorldChunk:

    def __init__(self, chunk_x: int, chunk_y: int, config: WorldConfig, *, seed: SeedType = None,
                 bit_length: int = 64) -> None:
        config.check()

        self._chunk_x = chunk_x
        self._chunk_y = chunk_y

        self._height_map = HeightMap(chunk_x, chunk_y, HeightMapConfig(config.chunk_size,
                                                                       config.height,
                                                                       config.roughness),
                                     seed=seed, bit_length=bit_length)
        if config.chunk_size != self._height_map.size:
            raise ValueError("Size mismatch")
        self.config = config

        self._seed = get_safe_seed(seed, bit_length)
        x = self._chunk_x * self.config.chunk_size
        y = self._chunk_y * self.config.chunk_size
        seed = (x ^ y << (bit_length >> 1)) ^ self._seed
        self._local_seed = seed & ((1 << bit_length) - 1)

    def generate(self) -> WorldChunkData:
        """Generates data for a chunk.

        :return: Contains the data generated.
        :rtype: :class:`.WorldChunkData`
        """
        height_map_image = self._height_map.generate()

        intensity_function = AdaptivePotentialFunction(self.config.chunk_size,
                                                       self.config.city_sizes)
        cities = [
            *MarkovChainMonteCarlo(
                MarkovChainMonteCarloIntensityFunction(
                    self.config.city_rate,
                    height_map_image,
                    intensity_function,
                    ExponentialZCompositeFunction()
                ),
                (
                    self.config.chunk_size,
                    self.config.chunk_size,
                    self.config.city_sizes + 1  # exclusive high
                ),
                self._local_seed
            )
        ]
        finder = Pathfinder(height_map_image, cities)
        paths = {
            key: path for i in range(len(cities)) for key, path in finder.shortest_paths(i).items()
        }

        world_data = WorldChunkData(
            self._chunk_x,
            self._chunk_y,
            height_map_image,
            cities,
            intensity_function.potential_map,
            paths
        )
        return world_data
