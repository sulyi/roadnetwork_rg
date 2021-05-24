"""Implementation of an I/O class for :class:`.WorldData` used by :class:`.WorldGenerator`"""

from __future__ import annotations

import hashlib
import hmac
import pickle
import struct
from io import BytesIO
from typing import BinaryIO, Callable, Dict, List, Tuple, TypeVar, Union

from PIL import Image

from .common import PixelPath, PointType, SeedType, WorldConfig, WorldChunkData, WorldData

T = TypeVar('T')


class DatafileDecodeError(Exception):
    """File corruption found."""


class DatafileEncodingError(Exception):
    """Unexpected data."""


class Datafile:
    """Implements an I/O file handler for :class:`.WorldData`.


    **Use cases**::

        Datafile.save(filename, key, data)
        data = Datafile.load(filename, key)

        file = Datafile()
        file.set_data(data)
        file.write(filename, key)

        file = Datafile()
        file.read(filename, key)
        data = file2.get_data()

    **File format**

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
    # TODO: add backward compatibility if needed

    __version = (0, 1, 1)
    __compatible_versions = ()
    __magic = '#D-MG#WG-D'.encode('ascii')
    __header_checksum_length = 64  # from `hashlib` `sha512` method `digest_size`
    __data_signature_length = 64  # from `hmac` `digest_size` with `sha512` method

    def __init__(self) -> None:
        self._data: bytes = b''

    def set_data(self, world: WorldData) -> None:
        """Sets binary data.

        Data can be to be saved by :meth:`write` or parsed by :meth:`get_data`.

        :param world: Data to be set.
        :type world: :class:`.WorldData`
        :raises: :exc:`TypeError`, if type of :attr:`world.seed` is not allowed.
            Only  :class:`None`, :class:`int`, :class:`str`, :class:`bytes`, and :class:`bytearray`
            are supported types
        :raises: :exc:`DatafileEncodingError`
        """

        # IDEA: use 2 bit `seed_type` instead byte?
        if world.seed is None:
            seed_type = 0
            seed = b''
        elif isinstance(world.seed, int):
            seed_type = 1
            seed = world.seed.to_bytes((world.seed.bit_length() + 7) // 8, byteorder='little')
        elif isinstance(world.seed, str):
            seed_type = 2
            seed = world.seed.encode('ascii')
        elif isinstance(world.seed, (bytes, bytearray)):
            seed_type = 3 if isinstance(world.seed, bytes) else 4
            seed = world.seed
        else:
            raise TypeError("Unsupported seed type of %s,\n"
                            "it should be either None, int, str, bytes, or bytearray" %
                            type(world.seed).__name__)
        seed = seed[:255]  # first 255 bytes (if larger)
        try:
            # FIXME: functions from `intensity`, use names only
            config = pickle.dumps([*world.config.__dict__.values()])
        except pickle.PicklingError as err:
            raise DatafileEncodingError("Failed to encode config") from err

        if len(seed) > 255:
            raise DatafileEncodingError("Too large seed")
        if len(config) > 255:
            raise DatafileEncodingError("Too large config")

        try:
            config_pack = struct.pack(
                '<B B %ds x Q x B %ds x H' % (len(seed), len(config)),
                seed_type,  # 1 byte
                len(seed),  # 1 byte
                seed,  # varied
                # pad 1 byte
                world.safe_seed,  # 8 bytes
                # pad 1 byte
                len(config),  # 1 byte
                config,  # 43 bytes
                # pad 1 byte
                len(world.chunks)  # 2 bytes
            )
        except struct.error as err:
            raise DatafileEncodingError("Failed to encode config") from err

        self._data = b'\00'.join((
            config_pack,
            # separator 1 byte
            b'\00'.join(self.encode_chunk(chunk) for chunk in world.chunks)
        ))

    def get_data(self) -> WorldData:
        """Parses binary data of file.

        Data can be loaded by :meth:`read` or set by :meth:`set_data`.

        :return: Parsed data.
        :rtype: :class:`.WorldData`
        """

        data = BytesIO(self._data)
        try:
            seed_type, seed_length = struct.unpack(
                '<B B',
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
            if seed_length != 0:
                raise DatafileDecodeError("Non-zero seed length of None seed")
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
                '<Q x B',
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

        chunks: List[WorldChunkData, ...] = [
            Datafile._read_list_item(data, i, chunks_count, b'\00', Datafile.decode_chunk)
            for i in range(chunks_count)
        ]

        return WorldData(config, seed, safe_seed, chunks)

    def clear_data(self) -> None:
        """Clears data set for :class:`Datafile` instance."""
        self._data = b''

    def write(self, filename: Union[str, bytes], key: bytes) -> None:
        """Writes data to a file, signs data using hashing algorithm.

        :param filename: It is the name of the destination to where data is saved.
        :type filename: :data:`~typing.Union` [:class:`str`, :class:`bytes` ]
        :param key: Hash key to be used signing data.
        :type key: :class:`bytes` (i.g. returned by :meth:`str.encode`)
        """

        if len(self._data) > 4294967295:
            raise DatafileEncodingError("Too large data")

        with open(filename, 'wb') as file:
            signature = hmac.new(key, self._data, hashlib.sha512).digest()
            # fail-safe (overkill)
            if len(signature) > self.__data_signature_length:
                raise DatafileEncodingError("Too large data signature")

            try:
                header = struct.pack(
                    '<L %ds' % self.__data_signature_length,
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
                    '<10s BBB %ds H' % self.__header_checksum_length,
                    self.__magic,  # 10 bytes
                    *Datafile.__version,  # 3 bytes
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
        """Reads data from file, authenticating data using hashing algorithm

        :param filename: It is the name of the source from where data is read.
        :type filename: :data:`~typing.Union` [:class:`str`, :class:`bytes`]
        :param key: Hash key to be used authenticating data.
        :type key: :class:`bytes` (i.g. returned by :meth:`str.encode`)
        """

        with open(filename, 'rb') as file:
            checksum, offset = self._read_magic(file)
            self._check_delimiter(file)
            content_length, signature = self._read_header(file, offset, checksum)

            self._check_delimiter(file)
            data = file.read(content_length)
            if signature != hmac.new(key, data, hashlib.sha512).digest():
                raise DatafileDecodeError("Invalid data signature")
            self._data = data

    @classmethod
    def get_version(cls) -> str:
        """Gets version of file format.

        :return: It is a string of the file format's version numbers separated by dots.
        :rtype: :class:`str`
        """

        return '.'.join(map(str, cls.__version))

    @classmethod
    def save(cls, filename: Union[str, bytes], key: bytes, world: WorldData) -> None:
        """Saves data to file.

        Creates an instance, sets data and writes it to destination.
        (see: :meth:`set_data` and :meth:`write`)
        """

        file = cls()
        file.set_data(world)
        file.write(filename, key)

    @classmethod
    def load(cls, filename: Union[str, bytes], key: bytes) -> Datafile:
        """Loads data from file.

        Creates an instance and reads data. (see: :meth:`read`)

        :return: It is the instance created.
        :rtype: :class:`Datafile`
        """

        instance = cls()
        instance.read(filename, key)
        return instance

    @staticmethod
    def encode_chunk(chunk: WorldChunkData) -> bytes:
        """Encodes a chunk.

        :param chunk: It is the object to be encoded.
        :type chunk: :class:`.WorldChunkData`
        :return: It is the encoded data.
        :rtype: :class:`bytes`
        """

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
                '<i i H %ds x H' % len(cities),
                chunk.offset_x,  # 4 bytes
                chunk.offset_y,  # 4 bytes
                len(cities),  # 2 bytes
                cities,  # varied
                # pad 1 byte
                len(chunk.pixel_paths)  # 2 bytes
            )
            height_map_pack = struct.pack(
                '<L %ds' % len(height_map),
                len(height_map),  # 2 bytes
                height_map  # varied
            )
            potential_map_pack = struct.pack(
                '<L %ds' % len(potential_map),
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
        """Reads data and decodes it.

        :param data: It is a buffer from data to be read.
        :type data: :class:`~typing.BinaryIO`
        :return: It is the decoded data.
        :rtype: :class:`.WorldChunkData`
        """

        try:
            offset_x: int
            offset_y: int
            offset_x, offset_y, cities_length = struct.unpack(
                '<i i H',
                data.read(10)
            )
        except struct.error as err:
            raise DatafileDecodeError(
                "Failed to decode offset_x, offset_y and/or length of chunk's cities attribute"
            ) from err

        try:
            cities: List[PointType, ...] = pickle.loads(data.read(cities_length))
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

        pixel_paths: Dict[Tuple[PointType, PointType], PixelPath] = dict(
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

        return WorldChunkData(offset_x, offset_y, height_map, cities, potential_map, pixel_paths)

    @staticmethod
    def encode_pixel_path(key: Tuple[PointType, PointType], path: PixelPath) -> bytes:
        """Encodes a key-value pair.

        :param key: It is a pair of points corresponding to the ends of *path*.
        :type key: :class:`tuple` [:data:`.PointType`, :data:`.PointType` ]
        :param path: It is a path between the tow point in *key*.
        :type path: :class:`.PixelPath`
        :return: It is the encoded data.
        :rtype: :class:`bytes`
        """

        try:
            pixels = pickle.dumps(path.pixels)
        except pickle.PicklingError as err:
            raise DatafileEncodingError("Failed to encode pixels") from err
        if len(pixels) > 65535:
            raise DatafileEncodingError("Too large pixels")

        source, target = key
        try:
            data = struct.pack(
                '<d HHH HHH H %ds' % len(pixels),
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
    def decode_pixel_path(data: BinaryIO) -> Tuple[Tuple[PointType, PointType], PixelPath]:
        """Reads data and decodes it.

        :param data: It is the buffer from data to be read.
        :type data: :class:`~typing.BinaryIO`
        :return: It is a *key-value* pair, *key* is a pair of points corresponding to the ends of
            the *value* which is a path between them.
        :rtype: :class:`tuple` [:class:`tuple` [:data:`.PointType`, :data:`.PointType` ],
            :class:`.PixelPath` ]
        """

        try:
            cost: float
            (cost,
             source_x, source_y, source_z,
             target_x, target_y, target_z,
             pixels_length
             ) = struct.unpack(
                '<d HHH HHH H',
                data.read(22)
            )
        except struct.error as err:
            raise DatafileDecodeError("Failed to decode"
                                      " PixelPath cost and length of pixels") from err
        try:
            pixels: List[Tuple[int, int], ...] = pickle.loads(data.read(pixels_length))
        except pickle.UnpicklingError as err:
            raise DatafileDecodeError("Failed to decode pixels") from err
        key: Tuple[PointType, PointType] = ((source_x, source_y, source_z),
                                            (target_x, target_y, target_z))
        return key, PixelPath(cost, pixels)

    @staticmethod
    def _read_list_item(data: BinaryIO, index: int, end: int, delimiter: bytes,
                        item_decoder: Callable[[BinaryIO], T]) -> T:
        item = item_decoder(data)
        # NOTE: there might be a better pattern
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
    def _read_magic(file: BinaryIO) -> Tuple[bytes, int]:
        compatible_versions = Datafile.__compatible_versions + (Datafile.__version,)
        try:
            (magic,
             vmajor, vminor, vpatch,
             checksum,
             offset) = struct.unpack(
                '<10s BBB %ds H' % Datafile.__header_checksum_length,
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
    def _read_header(file: BinaryIO, offset: int, checksum: bytes) -> Tuple[int, bytes]:
        header = file.read(offset)
        if checksum != hashlib.sha512(header).digest():
            raise DatafileDecodeError("Invalid headed checksum")
        try:
            content_length, signature = struct.unpack(
                '<L %ds' % Datafile.__data_signature_length,
                header
            )
        except struct.error as err:
            raise DatafileDecodeError("Failed to decode header") from err
        return content_length, signature
