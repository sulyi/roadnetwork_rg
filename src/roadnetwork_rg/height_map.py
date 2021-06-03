"""Terrain generator"""

import hashlib

from PIL import Image, ImageFilter, ImageChops

from .common import HeightMapConfig, SeedType, get_safe_seed


class HeightMap:
    """It is an implementation of diamond-square algorithm.

    In order to allow persistent map generation uses hash based pRNG.
    """

    k_diagonal = ImageFilter.Kernel((3, 3), [1, 0, 1, 0, 0, 0, 1, 0, 1])
    k_cross = ImageFilter.Kernel((3, 3), [0, 1, 0, 1, 0, 1, 0, 1, 0])

    def __init__(self, offset_x: int, offset_y: int, config: HeightMapConfig, *,
                 seed: SeedType = None, bit_length: int = 64) -> None:
        """Initializes height map generation.

        For more on parameters see also :class:`.HeightMapConfig`.

        :param offset_x: It is the map coordinate in the *x* axis.
        :type offset_x: :class:`int`
        :param offset_y: It is the map coordinate in the *y* axis.
        :type offset_y: :class:`int`
        :param config: It contains size, height and roughness.
        :type config: :class:`.HeightMapConfig`
        :param seed: It is used for ``xor`` ing joined coordinates before hashing.
        :type seed: :data:`.SeedType`
        :param bit_length: It is used for shifting *y* coordinate before hashing. Lower part is
            occupied by *x* coordinate.
        :type bit_length: :class:`int`
        """

        config.check()

        self._steps = (config.size - 1).bit_length()  # floor(log2(size - 1)) + 1 w/o math
        self._size = 1 << self._steps

        self._config = config
        self._bit_length = bit_length & ((1 << bit_length.bit_length()) - 2)  # make it even

        self._offset_x = offset_x
        self._offset_y = offset_y

        self._seed = get_safe_seed(seed, bit_length)

    @property
    def size(self) -> int:
        """It is the actual used size. May differ from :attr:`config.size`."""

        return self._size

    @property
    def bit_length(self) -> int:
        """It is the actual bit length used."""

        return self._bit_length

    # NOTE: hot-spot
    def _get_random_value(self, x: int, y: int) -> float:
        # uniform over [-1, 1] inclusive range
        at = (x ^ y << (self._bit_length >> 1)) ^ self._seed
        hashed = hashlib.sha256(at.to_bytes((self._bit_length >> 3) + 1, 'big', signed=True))
        max_value = (1 << (hashed.digest_size << 3)) - 1
        value = int.from_bytes(hashed.digest(), 'big') / max_value * 2 - 1
        return value

    def generate(self) -> Image.Image:
        """Generates a height map.

        :return: Generated height map.
        :rtype: :class:`PIL.Image.Image`
        """
        cx = self._offset_x * self._size
        cy = self._offset_y * self._size

        sub_size = self._size

        height = self._config.height * 127 - 1

        length = 2
        image = Image.frombytes('L', (length, length),
                                bytes(127 + int(self._get_random_value(x + cx, y + cy) *
                                                height)
                                      for x, y in ((0, 0), (sub_size, 0),
                                                   (0, sub_size), (sub_size, sub_size)))
                                )

        for _ in range(self._steps):
            height *= self._config.roughness

            length += length - 1
            sub_size >>= 1
            image = image.resize((length, length), resample=Image.LINEAR)
            # square step
            fim = image.filter(self.k_diagonal)
            displacement = [127] * (length * length)
            mask = [0] * (length * length)
            for i in range(1, length, 2):
                for j in range(1, length, 2):
                    value = self._get_random_value(i * sub_size + cx, j * sub_size + cy)
                    pixel = int(height * value) + 127
                    # fail-safe
                    if 0 < pixel < 255:
                        displacement[i + j * length] = pixel
                    elif pixel > 255:
                        displacement[i + j * length] = 255
                    else:
                        displacement[i + j * length] = 0
                    mask[i + j * length] = 255
            fim = ImageChops.add(fim, Image.frombytes('L', (length, length),
                                                      bytes(displacement)), offset=-127)
            mask = Image.frombytes('L', (length, length), bytes(mask))
            image.paste(fim, mask=mask)

            # diamond step
            fim = image.filter(self.k_cross)
            displacement = [127] * (length * length)
            mask = [0] * (length * length)
            for i in range(1, 2 * length, 2):
                for j in range(min(i + 1, length) - 1, max(0, i - length + 1) - 1, - 1):
                    value = self._get_random_value(j * sub_size + cx, (i - j) * sub_size + cy)
                    pixel = int(height * value) + 127
                    # fail-safe
                    if 0 < pixel < 255:
                        displacement[j + (i - j) * length] = pixel
                    elif pixel > 255:
                        displacement[j + (i - j) * length] = 255
                    else:
                        displacement[j + (i - j) * length] = 0
                    mask[j + (i - j) * length] = 255
            fim = ImageChops.add(fim, Image.frombytes('L', (length, length),
                                                      bytes(displacement)), offset=-127)
            mask = Image.frombytes('L', (length, length), bytes(mask))
            image.paste(fim, mask=mask)

        return image.crop((0, 0, self._size, self._size))
