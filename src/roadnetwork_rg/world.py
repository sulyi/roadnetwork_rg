"""Implementation of handlers for generating the world"""

from __future__ import annotations

import warnings
from typing import Dict, List, Set, Tuple, Union

from PIL import Image, ImageDraw, ImageStat

from .common import (HeightMapConfig, PointType, SeedType, WorldChunkData, WorldConfig,
                     WorldData, WorldRenderOptions, get_safe_seed)
from .data import colour_palette
from .datafile import Datafile
from .height_map import HeightMap
from .heuristic import greedy
from .intensity import (AdaptivePotentialFunction, ExponentialZCompositeFunction,
                        MarkovChainMonteCarloPotentialFunction,
                        MarkovChainMonteCarloIntensityFunction)
from .pathfinder import Pathfinder
from .point_process import IntensityFunction, MarkovChainMonteCarlo, PointProcess

default_world_config = WorldConfig(
    chunk_size=256, height=1., roughness=.5, city_rate=32,
    intensity_function=MarkovChainMonteCarloIntensityFunction.__name__,
    potential_function=AdaptivePotentialFunction.__name__,
    point_process=MarkovChainMonteCarlo.__name__,
    filter=greedy.__name__,
    city_sizes=8
)
"""It is the default configuration for :class:`WorldGenerator` initialization."""
default_render_options = WorldRenderOptions()
"""It is the default render options for :meth:`WorldGenerator.render` method."""


def config_factory(
        chunk_size: int = None,
        height: float = None,
        roughness: float = None,
        city_rate: int = None,
        intensity_function: str = None,
        potential_function: str = None,
        point_process: str = None,
        heuristic: str = None,
        city_sizes: int = None,
        bit_length: int = None
) -> WorldConfig:
    chunk_size = chunk_size if chunk_size is not None else default_world_config.chunk_size
    height = height if height is not None else default_world_config.height
    roughness = roughness if roughness is not None else default_world_config.roughness
    city_rate = city_rate if city_rate is not None else default_world_config.city_rate
    intensity_function = intensity_function if intensity_function is not None \
        else default_world_config.intensity_function
    potential_function = potential_function if potential_function is not None \
        else default_world_config.potential_function
    point_process = point_process if point_process is not None \
        else default_world_config.point_process
    city_sizes = city_sizes if city_sizes is not None else default_world_config.city_sizes
    heuristic = heuristic if heuristic is not None else default_world_config.filter
    bit_length = bit_length if bit_length is not None else default_world_config.bit_length

    return WorldConfig(chunk_size, height, roughness, city_rate,
                       intensity_function, potential_function, point_process,
                       heuristic, city_sizes, bit_length)


class WorldDataInconsistencyError(Exception):
    """WorldData is unsafe"""


class WorldGenerator:
    """It is a handler to generate a map with terrain, cities and roads."""

    __city_r = 2
    __city_colour = (255, 0, 0)
    __text_color = (0, 0, 0)
    __city_border = (0, 0, 0)

    def __init__(self, *, config: WorldConfig = default_world_config,
                 seed: SeedType = None) -> None:
        """Initializes the generator.

        :param config: It is the configuration used for setting various variables.
        :type config: :class:`.WorldConfig`
        :param seed: It is the seed used for initializing pRNG.
        :type seed: :class:`int`
        """

        config.check()
        self._config = config
        self._chunks: Dict[Tuple[int, int], WorldChunkData] = {}
        self._selected_paths: Dict[Tuple[int, int], Set[Tuple[PointType, PointType], ...]] = {}

        # for i/o compatibility truncate seed to 255
        self._seed = seed if isinstance(seed, int) or seed is None else seed[:255]
        self._safe_seed = get_safe_seed(seed, self._config.bit_length)

    @classmethod
    def save(cls, instance: WorldGenerator, filename: Union[str, bytes], key: bytes) -> None:
        """Saves generated data to file.

        see also: :meth:`WorldGenerator.write`
        """

        if not isinstance(instance, cls):
            raise ValueError("Argument is not an %s object" % cls.__name__)
        instance.write(filename, key)

    @classmethod
    def load(cls, filename: Union[str, bytes], key: bytes) -> WorldGenerator:
        """Loads data form a file.

        see also: :meth:`WorldGenerator.read`

        :return: An object containing loaded data.
        :rtype: :class:`WorldGenerator`
        """

        instance = cls.__new__(cls)
        instance.read(filename, key)
        return instance

    @property
    def config(self):
        """It is the generator's configuration."""

        return self._config

    @property
    def seed(self) -> SeedType:
        """It is the seed set or if it is :data:`None` the safe seed actually used by pRNG."""

        return self._seed if self._seed is not None else self._safe_seed

    def get_chunks(self) -> List[WorldChunkData, ...]:
        """Provides a list of chunks.

        :returns: A copy of the chunks generated.
        :rtype: :class:`list` [:class:`.WorldChunkData`, ...]
        """

        return list(self._chunks.values())

    def read(self, filename: Union[str, bytes], key: bytes) -> None:
        """Reads data form a file.

        :param filename: It is the source from where data is loaded.
        :type filename: :data:`~typing.Union` [:class:`str`, :class:`bytes`]
        :param key: It is a hashing key used for validating data.
        :type key: :class:`bytes`
        """
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

        filter_function = globals().get(self.config.filter)
        # FIXME: type check
        if filter_function is None:
            raise ValueError("Unknown name, '%s', for potential function"
                             % self.config.filter)
        self._selected_paths = {
            (chunk.offset_x, chunk.offset_y): filter_function(chunk.cities, chunk.pixel_paths)
            for chunk in data.chunks
        }

        self._seed = seed
        self._safe_seed = data.safe_seed

        if dirty:
            raise WorldDataInconsistencyError("Error loading file")

    def write(self, filename: Union[str, bytes], key: bytes) -> None:
        """Writes generated data to file.

        :param filename: It is the destination where data is saved.
        :type filename: :data:`~typing.Union` [:class:`str`, :class:`bytes`]
        :param key: It is a hashing key used for signing data.
        :type key: :class:`bytes`
        """

        Datafile.save(filename, key,
                      WorldData(self._config, self._seed, self._safe_seed, self.get_chunks()))

    def add_chunk(self, chunk_x: int, chunk_y: int) -> None:
        """Generates a new chunk.

        :param chunk_x: It is the **x** coordinate of the new chunk.
        :type chunk_x: :class:`int`
        :param chunk_y: It is the **y** coordinate of the new chunk.
        :type chunk_y: :class:`int`
        :raises: :exc:`KeyError` If chunk already exists.
        """

        if (chunk_x, chunk_y) in self._chunks:
            warnings.warn("Chunk already exists")
            return

        chunk = WorldChunk(chunk_x, chunk_y, self._config, seed=self._safe_seed,
                           bit_length=self._config.bit_length)
        chunk_data = chunk.generate()

        filter_function = globals().get(self.config.filter)
        # FIXME: type check
        if filter_function is None:
            raise ValueError("Unknown name, '%s', for potential function"
                             % self.config.filter)

        self._selected_paths[chunk_x, chunk_y] = filter_function(chunk_data.cities,
                                                                 chunk_data.pixel_paths)

        self._chunks[chunk_x, chunk_y] = chunk_data

    def render(self, *, options: WorldRenderOptions = default_render_options) -> Image.Image:
        """Create an image from generated data depending on options`.

        :param options: Defines how data is rendered, defaults to :data:`default_render_options`.
        :type options: :class:`.WorldRenderOptions`
        :return: :class:`PIL.Image.Image`
        """

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

        draw = ImageDraw.Draw(atlas_im)

        for key, chunk in self._chunks.items():
            cx = (chunk.offset_x - min_x) * self._config.chunk_size
            cy = (chunk.offset_y - min_y) * self._config.chunk_size

            # concatenate height maps
            if options.show_height_map:
                if options.colour_height_map:
                    image = chunk.height_map.convert('P')
                    image.putpalette(colour_palette)
                    atlas_im.paste(image, (cx, cy))
                else:
                    atlas_im.paste(chunk.height_map, (cx, cy))

            # overlay potential field
            if options.show_potential_map:
                image = Image.new('RGBA', chunk.potential_map.size, 0)
                image.putalpha(chunk.potential_map)
                atlas_im.alpha_composite(image, (cx, cy))

            # draw roads
            if options.show_roads:
                image = self._render_draw_roads(chunk, self._selected_paths[key])
                atlas_im.paste(image, (cx, cy), mask=image)

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

        return atlas_im

    @staticmethod
    def _render_draw_roads(chunk: WorldChunkData,
                           selected_paths: Set[Tuple[PointType, PointType]]) -> Image.Image:
        # NOTE: avoiding `Image.Image.putpixel`
        path_data = [0] * (chunk.height_map.size[0] * chunk.height_map.size[1])
        for path in selected_paths:
            for point_x, point_y in chunk.pixel_paths[path].pixels:
                path_data[point_x + point_y * chunk.height_map.size[0]] = 255
        image = Image.new('RGBA', chunk.height_map.size, 0)
        image.putalpha(Image.frombytes('L', chunk.height_map.size,
                                       bytes(path_data)))
        return image


class WorldChunk:
    """It is a handler to generate a new chunk."""

    def __init__(self, chunk_x: int, chunk_y: int, config: WorldConfig, *, seed: SeedType = None,
                 bit_length: int = 64) -> None:
        """Initializes a new chunk.

        :param chunk_x: It is the **x** coordinate of the new chunk.
        :type chunk_x: :class:`int`
        :param chunk_y: It is the **y** coordinate of the new chunk.
        :type chunk_y: :class:`int`
        :param config: It is the configuration provided by :class:`WorldGenerator`.
        :type config: :class:`.WorldConfig`
        :param seed: It is the seed used for initializing pRNG.
        :type seed: :class:`int`
        :param bit_length: See: :class:`.HeightMap`.
        :type bit_length: :class:`int`
        """
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

        intensity_function_class: IntensityFunction.__class__
        potential_function_class: MarkovChainMonteCarloPotentialFunction.__class__
        point_process_class: PointProcess.__class__

        intensity_function_class, potential_function_class, point_process_class = (
            globals().get(self.config.intensity_function),
            globals().get(self.config.potential_function),
            globals().get(self.config.point_process)
        )

        # FIXME: type check
        if potential_function_class is None:
            raise ValueError("Unknown name, '%s', for potential function"
                             % self.config.potential_function)
        if intensity_function_class is None:
            raise ValueError("Unknown name, '%s', for intensity function"
                             % self.config.intensity_function)
        if point_process_class is None:
            raise ValueError("Unknown name, '%s', for point process"
                             % self.config.point_process)

        potential_function = potential_function_class(self.config.chunk_size,
                                                      self.config.city_sizes)
        cities = [
            *point_process_class(
                intensity_function_class(
                    self.config.city_rate,
                    height_map_image,
                    potential_function,
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
            potential_function.potential_map,
            paths
        )
        return world_data
