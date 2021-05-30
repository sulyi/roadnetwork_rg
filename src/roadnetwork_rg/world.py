from __future__ import annotations

import hashlib
import hmac
import pickle
import struct
import warnings
from dataclasses import dataclass, field
from io import BytesIO
from typing import Union, Callable, TypeVar, BinaryIO

from PIL import Image, ImageDraw, ImageStat

from . import __version_info__ as package_version
from .common import get_safe_seed, SeedType, PointType
from .data import colour_palette
from .height_map import HeightMap
from .intensity import AdaptivePotentialFunction, ExponentialZCompositeFunction, \
    MarkovChainMonteCarloIntensityFunction
from .pathfinder import find_shortest_paths, PixelPath
from .point_process import MarkovChainMonteCarlo

T = TypeVar('T')


@dataclass(frozen=True)
class WorldConfig:
    chunk_size: int
    height: float
    roughness: float
    city_rate: int
    city_sizes: int = 1
    bit_length: int = 64

    def check(self) -> None:
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
    pixel_paths: dict[tuple[PointType, PointType], PixelPath] = field(compare=False)


default_world_config = WorldConfig(chunk_size=256, height=1., roughness=.5,
                                   city_rate=32, city_sizes=8)
default_render_options = WorldRenderOptions()


class DatafileDecodeError(Exception):
    """File corruption found"""


class DatafileEncodingError(Exception):
    """Unexpected data"""


class Datafile:
    """file format:

    * all numbers are in little endian

    ============== =================
        size        description
    ============== =================
       10 bytes      magic number
        3 bytes     version number
       64 bytes      header checksum
        2 bytes       data offset
     -- 1 byte --   -- separator --
     header:
        4 bytes         content length (4Gb data max)
       64 bytes         signature
     -- 1 byte --   -- separator --
     data:
        1 byte         seed type (None:0, int:1, str:2, bytes:3, bytearray:4)
        1 byte        length of seed
        varied            seed
     -- 1 byte --   --    pad    --
        8 bytes        safe_seed
     -- 1 byte --   --    pad    --
        1 byte      length of config (future proof, currently 43)
        varied           config (pickled format)
     -- 1 byte --   --    pad    --
     chunks:
        2 bytes     number of chunks
     -- 1 byte --   -- separator --
     chunk:
        4 bytes           x
        4 bytes           y
        2 bytes     length of cities
        varied          cities (pickled format)
     -- 1 byte --   --    pad    --
        2 bytes     number of paths
     -- 1 byte --   -- separator --
     pixel_path:
        8 bytes          cost
        6 byte          source (2, 2, 2 byte)
        6 byte          target (2, 2, 2 byte)
        2 bytes     length of pixels
        varied          pixels (pickled format)
     -- 1 byte --   -- separator --
        4 bytes     length of height_map
        varied         height_map (in TIFF format)
     -- 1 byte --   -- separator --
        4 bytes     length of potential_map
        varied        potential_map (in TIFF format)
     -- 1 byte --   -- separator -- (if not last)
    ============== =================

    """

    # TODO: decide
    # if EOF is needed, (note signature)
    # if header needs other things
    # if 2 bytes of short is enough for length of cities (note when pickled)
    # if 2 bytes of short is enough for length of pixels (note when pickled)

    __compatible_versions = ((0, 1, 0),)
    __magic = '#D-MG#WG-D'.encode('ascii')
    __header_checksum_length = 64  # from `hashlib` `sha512` method `digest_size`
    __data_signature_length = 64  # from `hmac` `digest_size` with `sha512` method

    def __init__(self) -> None:
        self._data: bytes = b''

    @classmethod
    def save(cls, filename: Union[str, bytes], key: bytes, world: WorldGenerator) -> None:
        file = cls()
        file.set_data(world.config, world.actual_seed, world.safe_seed, world.get_chunks())
        file.write(filename, key)

    @classmethod
    def load(cls, filename: Union[str, bytes], key: bytes) -> Datafile:
        instance = cls()
        instance.read(filename, key)
        return instance

    def set_data(self, config: WorldConfig, seed: SeedType, safe_seed: int,
                 chunks: list[WorldChunkData, ...]) -> None:
        # IDEA: use 2 bit `seed_type` instead byte?
        if seed is None:
            seed_type = 0
            seed = b''
        elif isinstance(seed, int):
            seed_type = 1
            seed = seed.to_bytes((seed.bit_length() + 7) // 8, byteorder='little')
        elif isinstance(seed, str):
            seed_type = 2
            seed = seed.encode('ascii')
        elif isinstance(seed, bytes):
            seed_type = 3
        else:
            seed_type = 4
        seed = seed[:255]  # first 255 bytes (if larger)
        try:
            config = pickle.dumps(list(config.__dict__.values()))
        except pickle.PicklingError as err:
            raise DatafileEncodingError("Failed to encode config") from err

        if len(seed) > 255:
            raise DatafileEncodingError("Too large seed")
        if len(config) > 255:
            raise DatafileEncodingError("Too large config")

        try:
            config_pack = struct.pack(
                '<BB%dsxQxB%dsxH' % (len(seed), len(config)),
                seed_type,  # 1 byte
                len(seed),  # 1 byte
                seed,  # varied
                # pad 1 byte
                safe_seed,  # 8 bytes
                # pad 1 byte
                len(config),  # 1 byte
                config,  # 43 bytes
                # pad 1 byte
                len(chunks)  # 2 bytes
            )
        except struct.error as err:
            raise DatafileEncodingError("Failed to encode config") from err

        self._data = b'\00'.join((
            config_pack,
            # separator 1 byte
            b'\00'.join(self.encode_chunk(chunk) for chunk in chunks)
        ))

    def get_data(self) -> tuple[SeedType, int, WorldConfig, list[WorldChunkData, ...]]:
        data = BytesIO(self._data)
        try:
            seed_type, seed_length = struct.unpack(
                '<BB',
                data.read(2)
            )
        except struct.error as err:
            raise DatafileDecodeError("Failed to decode  seed type and length") from err

        try:
            seed, = struct.unpack(
                '<%ds' % seed_length,
                data.read(seed_length)
            )
        except struct.error as err:
            raise DatafileDecodeError("Failed to decode seed") from err

        seed: SeedType
        if seed_type == 0:  # None
            seed = None
        elif seed_type == 1:  # int
            seed = int.from_bytes(seed, 'little')
        else:
            if seed_type == 2:  # str
                seed: str = seed.decode('ascii')
            elif seed_type == 3:  # bytes
                pass
            elif seed_type == 4:  # bytearray
                seed: bytearray = bytearray(seed)
            else:
                raise DatafileDecodeError("Unrecognised seed type")

        self._check_delimiter(data)

        try:
            safe_seed: int
            safe_seed, config_length = struct.unpack(
                '<QxB',
                data.read(10)
            )
        except struct.error as err:
            raise DatafileDecodeError("Failed to decode safe_seed and length of config") from err

        try:
            config: WorldConfig = WorldConfig(*pickle.loads(data.read(config_length)))
        except pickle.UnpicklingError as err:
            raise DatafileDecodeError("Failed to decode config") from err

        self._check_delimiter(data)

        try:
            chunks_count, = struct.unpack(
                '<H',
                data.read(2)
            )
        except struct.error as err:
            raise DatafileDecodeError("Failed to decode length of chunks") from err

        self._check_delimiter(data)

        chunks: list[WorldChunkData, ...] = [
            Datafile._read_list_item(data, i, chunks_count, b'\00', Datafile.decode_chunk)
            for i in range(chunks_count)
        ]

        return seed, safe_seed, config, chunks

    def clear_data(self) -> None:
        self._data = b''

    def write(self, filename: Union[str, bytes], key: bytes) -> None:
        if len(self._data) > 4294967295:
            raise DatafileEncodingError("Too large data")

        with open(filename, 'wb') as file:
            signature = hmac.new(key, self._data, hashlib.sha512).digest()
            # fail-safe (overkill)
            if len(signature) > self.__data_signature_length:
                raise DatafileEncodingError("Too large data signature")

            try:
                header = struct.pack(
                    '<L%ds' % self.__data_signature_length,
                    len(self._data),  # 4 bytes
                    signature,  # 64 bytes
                )
            except struct.error as err:
                raise DatafileEncodingError("Failed to encode header") from err

            checksum = hashlib.sha512(header).digest()
            # fail-safe (overkill)
            if len(checksum) > self.__header_checksum_length:
                raise DatafileEncodingError("Too large header checksum")

            try:
                magic = struct.pack(
                    '<10s3B%dsH' % self.__header_checksum_length,
                    self.__magic,  # 10 bytes
                    *package_version,  # 3 bytes
                    checksum,  # 64 bytes
                    len(header),  # 2 bytes
                )
            except struct.error as err:
                raise DatafileEncodingError("Failed to encode magic") from err

            file.write(b'\00'.join((
                magic,
                # separator 1 byte
                header,
                # separator 1 byte
                self._data
            )))

    def read(self, filename: Union[str, bytes], key: bytes) -> None:
        with open(filename, 'rb') as file:
            checksum, offset = self._read_magick(file)
            self._check_delimiter(file)
            content_length, signature = self._read_header(file, offset, checksum)

            self._check_delimiter(file)
            data = file.read(content_length)
            if signature != hmac.new(key, data, hashlib.sha512).digest():
                raise DatafileDecodeError("Invalid data signature")
            self._data = data

    @staticmethod
    def encode_chunk(chunk: WorldChunkData) -> bytes:
        try:
            cities = pickle.dumps(chunk.cities)
        except pickle.PicklingError as err:
            raise DatafileEncodingError("Failed to encode cities") from err
        if len(cities) > 65535:
            raise DatafileEncodingError("Too large cities")

        with BytesIO() as buff:
            chunk.height_map.save(buff, format='TIFF')
            height_map = buff.getvalue()
        if len(height_map) > 4294967295:
            raise DatafileEncodingError("Too large height_map")

        with BytesIO() as buff:
            chunk.potential_map.save(buff, format='TIFF')
            potential_map = buff.getvalue()
        if len(height_map) > 4294967295:
            raise DatafileEncodingError("Too large potential_map")

        try:
            chunk_pack = struct.pack(
                '<iiH%dsxH' % len(cities),
                chunk.x,  # 4 bytes
                chunk.y,  # 4 bytes
                len(cities),  # 2 bytes
                cities,  # varied
                # pad 1 byte
                len(chunk.pixel_paths)  # 2 bytes
            )
            height_map_pack = struct.pack(
                '<L%ds' % len(height_map),
                len(height_map),  # 2 bytes
                height_map  # varied
            )
            potential_map_pack = struct.pack(
                '<L%ds' % len(potential_map),
                len(potential_map),  # 2 bytes
                potential_map  # varied
            )
        except struct.error as err:
            raise DatafileEncodingError("Failed to encode chunk") from err

        data = b'\00'.join((
            chunk_pack,
            # separator 1 byte
            b'\00'.join(Datafile.encode_pixel_path(key, path)
                        for key, path in chunk.pixel_paths.items()),
            # separator 1 byte
            height_map_pack,
            # separator 1 byte
            potential_map_pack
        ))
        return data

    @staticmethod
    def decode_chunk(data: BinaryIO) -> WorldChunkData:
        try:
            x: int
            y: int
            x, y, cities_length = struct.unpack(
                '<iiH',
                data.read(10)
            )
        except struct.error as err:
            raise DatafileDecodeError("Failed to decode x, y, length of cities of chunk") from err

        try:
            cities: list[PointType, ...] = pickle.loads(data.read(cities_length))
        except pickle.UnpicklingError as err:
            raise DatafileDecodeError("Failed to decode cities") from err

        Datafile._check_delimiter(data)

        try:
            pixel_paths_count, = struct.unpack(
                '<H',
                data.read(2)
            )
        except struct.error as err:
            raise DatafileDecodeError("Failed to decode length of pixel_paths") from err

        Datafile._check_delimiter(data)

        pixel_paths: dict[tuple[PointType, PointType], PixelPath] = dict(
            Datafile._read_list_item(data, i, pixel_paths_count, b'\00', Datafile.decode_pixel_path)
            for i in range(pixel_paths_count)
        )

        Datafile._check_delimiter(data)

        try:
            height_map_length, = struct.unpack(
                '<L',
                data.read(4)
            )
            height_map: Image.Image = Image.open(BytesIO(data.read(height_map_length)))
        except struct.error as err:
            raise DatafileDecodeError("Failed to decode length of height_map") from err

        Datafile._check_delimiter(data)

        try:
            potential_map_length, = struct.unpack(
                '<L',
                data.read(4)
            )
        except struct.error as err:
            raise DatafileDecodeError("Failed to decode length of potential_map") from err

        potential_map: Image.Image = Image.open(BytesIO(data.read(potential_map_length)))

        return WorldChunkData(x, y, height_map, cities, potential_map, pixel_paths)

    @staticmethod
    def encode_pixel_path(key: tuple[PointType, PointType], path: PixelPath) -> bytes:
        try:
            pixels = pickle.dumps(path.pixels)
        except pickle.PicklingError as err:
            raise DatafileEncodingError("Failed to encode pixels") from err
        if len(pixels) > 65535:
            raise DatafileEncodingError("Too large pixels")

        source, target = key
        try:
            data = struct.pack(
                '<d HHH HHH H%ds' % len(pixels),
                path.cost,  # 8 bytes
                *source,
                *target,
                len(pixels),  # 2 bytes
                pixels  # varied
            )
        except struct.error as err:
            raise DatafileEncodingError("Failed to encode PixelPath") from err

        return data

    @staticmethod
    def decode_pixel_path(data: BinaryIO) -> tuple[tuple[PointType, PointType], PixelPath]:
        try:
            cost: float
            (cost,
             source_x, source_y, source_z,
             target_x, target_y, target_z,
             pixels_length) = struct.unpack(
                '<d HHH HHH H',
                data.read(22)
            )
        except struct.error as err:
            raise DatafileDecodeError("Failed to decode"
                                      " PixelPath cost and length of pixels") from err
        try:
            pixels: list[tuple[int, int], ...] = pickle.loads(data.read(pixels_length))
        except pickle.UnpicklingError as err:
            raise DatafileDecodeError("Failed to decode pixels") from err
        key: tuple[PointType, PointType] = ((source_x, source_y, source_z),
                                            (target_x, target_y, target_z))
        return key, PixelPath(cost, pixels)

    @staticmethod
    def _read_list_item(data: BinaryIO, index: int, end: int, delimiter: bytes,
                        item_decoder: Callable[[BinaryIO], T]) -> T:
        item = item_decoder(data)
        # XXX: there might be a better pattern
        # if there is a surrounding `delimiter` it could be handled
        # here, or before the `_decode_pixel_path` call
        if index < end - 1:
            if data.read(len(delimiter)) != delimiter:
                raise DatafileDecodeError
        return item

    @staticmethod
    def _check_delimiter(data: BinaryIO) -> None:
        if data.read(1) != b'\00':
            raise DatafileDecodeError

    @staticmethod
    def _read_magick(file: BinaryIO) -> tuple[bytes, int]:
        compatible_versions = (package_version, *Datafile.__compatible_versions)
        try:
            magic, vmajor, vminor, vpatch, checksum, offset = struct.unpack(
                '<10s3B%dsH' % Datafile.__header_checksum_length,
                file.read(79)
            )
        except struct.error as err:
            raise DatafileDecodeError("Failed to decode magic") from err
        if magic != Datafile.__magic:
            raise DatafileDecodeError("Argument filename has unrecognised format")
        if (vmajor, vminor, vpatch) not in compatible_versions:
            raise DatafileDecodeError("Argument filename has a non-compatible version")
        return checksum, offset

    @staticmethod
    def _read_header(file: BinaryIO, offset: int, checksum: bytes) -> tuple[int, bytes]:
        header = file.read(offset)
        if checksum != hashlib.sha512(header).digest():
            raise DatafileDecodeError("Invalid headed checksum")
        try:
            content_length, signature = struct.unpack(
                '<L%ds' % Datafile.__data_signature_length,
                header
            )
        except struct.error as err:
            raise DatafileDecodeError("Failed to decode header") from err
        return content_length, signature


class WorldGenerator:
    __city_r = 2
    __city_colour = (255, 0, 0)
    __text_color = (0, 0, 0)
    __city_border = (0, 0, 0)

    def __init__(self, *, config: WorldConfig = default_world_config,
                 seed: SeedType = None) -> None:
        config.check()
        self.config = config
        self._chunks: list[WorldChunkData, ...] = []

        # for i/o compatibility truncate seed to 255
        self._seed = seed if isinstance(seed, int) or seed is None else seed[:255]
        self._safe_seed = get_safe_seed(seed, self.config.bit_length)

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
    def seed(self) -> SeedType:
        return self._seed if self._seed is not None else self._safe_seed

    @property
    def actual_seed(self) -> SeedType:
        return self._seed

    @property
    def safe_seed(self) -> int:
        return self._safe_seed

    def get_chunks(self) -> list[WorldChunkData, ...]:
        return self._chunks.copy()

    def read(self, filename: Union[str, bytes], key: bytes) -> None:
        seed, safe_seed, config, chunks = Datafile.load(filename, key).get_data()
        try:
            config.check()
        except (ValueError, TypeError) as err:
            config = default_world_config
            warnings.warn("Failed `config` check, set to default, due\n\t:%s" % err)
        if seed is not None and get_safe_seed(seed, config.bit_length) == safe_seed:
            seed = None
            warnings.warn("Mismatching seed and safe_seed, seed is discarded")
        self.config = config
        self._chunks = chunks

        self._seed = seed
        self._safe_seed = safe_seed

    def write(self, filename: Union[str, bytes], key: bytes) -> None:
        Datafile.save(filename, key, self)

    def add_chunk(self, chunk_x: int, chunk_y: int) -> None:
        chunk = WorldChunk(chunk_x, chunk_y, self.config, seed=self._safe_seed,
                           bit_length=self.config.bit_length)
        chunk_data = chunk.generate()
        self._chunks.append(chunk_data)

    def render(self, *, options: WorldRenderOptions = default_render_options) -> Image.Image:
        if not self._chunks:
            raise IndexError("There are no chunks added to render")
        if not any((options.show_debug, options.show_height_map, options.show_cities,
                    options.show_roads, options.show_potential_map)):
            raise ValueError("Nothing to render with given 'option' argument")

        width = (max(self._chunks, key=lambda item: item.x).x -
                 min(self._chunks, key=lambda item: item.x).x + 1) * self.config.chunk_size
        height = (max(self._chunks, key=lambda item: item.y).y -
                  min(self._chunks, key=lambda item: item.y).y + 1) * self.config.chunk_size

        atlas_im = Image.new('RGBA', (width, height))
        draw_im = Image.new('RGBA', (width, height))

        draw = ImageDraw.Draw(draw_im)

        for chunk in self._chunks:
            cx = (chunk.x - min(self._chunks, key=lambda item: item.x).x) * self.config.chunk_size
            cy = (chunk.y - min(self._chunks, key=lambda item: item.y).y) * self.config.chunk_size

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
                msg = '\n'.join((
                    f"count: {len(chunk.cities)}",
                    f"expected: {self.config.city_rate}",
                    f"mean: {(255 - ImageStat.Stat(chunk.height_map).mean.pop()) / 255:.3f}",
                    f"sizes: {self.config.city_sizes}")
                )
                draw.multiline_text((cx, cy), msg, fill=self.__text_color)

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
        self.check(config.city_rate, config.city_sizes)

        self._chunk_x = chunk_x
        self._chunk_y = chunk_y

        self._height_map = HeightMap(chunk_x, chunk_y, config.chunk_size,
                                     config.height, config.roughness, seed=seed,
                                     bit_length=bit_length)
        if config.chunk_size != self._height_map.size:
            raise ValueError("Size mismatch")
        self.config = config

        self._seed = get_safe_seed(seed, bit_length)
        x = self._chunk_x * self.config.chunk_size
        y = self._chunk_y * self.config.chunk_size
        seed = (x ^ y << (bit_length >> 1)) ^ self._seed
        self._local_seed = seed & ((1 << bit_length) - 1)

    @staticmethod
    def check(city_rate: int, city_sizes: int) -> None:
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

    @property
    def size(self) -> int:
        return self.config.chunk_size

    @property
    def height_map(self) -> HeightMap:
        return self._height_map

    def generate(self) -> WorldChunkData:
        height_map_image = self.height_map.generate()

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

        paths = {}
        for i, source in enumerate(cities[:-1], 1):
            paths.update(find_shortest_paths(height_map_image, source, cities[i:]))

        world_data = WorldChunkData(
            self._chunk_x,
            self._chunk_y,
            height_map_image,
            cities,
            intensity_function.potential_map,
            paths
        )
        return world_data
