import hashlib

from PIL import Image, ImageFilter, ImageChops

from .common import get_safe_seed, SeedType


class HeightMap:

    def __init__(self, offset_x: int, offset_y: int, size: int, height: float, roughness: float, *,
                 seed: SeedType = None, bit_length: int = 64) -> None:
        self.check(size, height, roughness)

        self._steps = size.bit_length() - 1  # log2 w/o math.log2
        self._size = 1 << self._steps

        self._roughness = roughness
        self._height = height
        self._bit_length = bit_length >> 1 << 1  # make it even

        self._offset_x = offset_x
        self._offset_y = offset_y

        self._seed = get_safe_seed(seed, bit_length)

    @staticmethod
    def check(size: int, height: float, roughness: float) -> None:
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

    @property
    def size(self) -> int:
        return self._size

    @property
    def bit_length(self) -> int:
        return self._bit_length

    # XXX: hot-spot
    def _get_random_value(self, x, y) -> float:
        # uniform over [-1, 1] inclusive range
        at = (x ^ y << (self._bit_length >> 1)) ^ self._seed
        h = hashlib.sha256(at.to_bytes((self._bit_length >> 3) + 1, 'big', signed=True))
        max_value = (1 << (h.digest_size << 3)) - 1
        value = int.from_bytes(h.digest(), 'big') / max_value * 2 - 1
        return value

    def generate(self) -> Image.Image:
        cx = self._offset_x * self._size
        cy = self._offset_y * self._size

        sub_size = self._size

        height = self._height * 127 - 1

        length = 2
        image = Image.frombytes('L', (length, length),
                                bytes(127 + int(self._get_random_value(x + cx, y + cy) * height)
                                      for x, y in ((0, 0), (sub_size, 0),
                                                   (0, sub_size), (sub_size, sub_size)))
                                )

        k_diagonal = ImageFilter.Kernel((3, 3), [1, 0, 1, 0, 0, 0, 1, 0, 1])
        k_cross = ImageFilter.Kernel((3, 3), [0, 1, 0, 1, 0, 1, 0, 1, 0])

        for _ in range(self._steps):
            height *= self._roughness

            # square step
            length += length - 1
            sub_size >>= 1
            image = image.resize((length, length), resample=Image.LINEAR)
            fim = image.filter(k_diagonal)

            r = [127] * (length * length)
            mask = [0] * (length * length)

            for i in range(1, length, 2):
                for j in range(1, length, 2):
                    value = self._get_random_value(i * sub_size + cx, j * sub_size + cy)
                    pixel = int(height * value) + 127
                    # fail-safe
                    if 0 < pixel < 255:
                        r[i + j * length] = pixel
                    elif pixel > 255:
                        r[i + j * length] = 255
                    else:
                        r[i + j * length] = 0
                    mask[i + j * length] = 255

            fim = ImageChops.add(fim, Image.frombytes('L', (length, length), bytes(r)), offset=-127)
            image.paste(fim, mask=Image.frombytes('L', (length, length), bytes(mask)))

            # diamond step
            fim = image.filter(k_cross)

            r = [127] * (length * length)
            mask = [0] * (length * length)

            for p in range(1, 2 * length, 2):
                for q in range(min(p + 1, length) - 1, max(0, p - length + 1) - 1, - 1):
                    value = self._get_random_value(q * sub_size + cx, (p - q) * sub_size + cy)
                    pixel = int(height * value) + 127
                    # fail-safe
                    if 0 < pixel < 255:
                        r[q + (p - q) * length] = pixel
                    elif pixel > 255:
                        r[q + (p - q) * length] = 255
                    else:
                        r[q + (p - q) * length] = 0
                    mask[q + (p - q) * length] = 255

            fim = ImageChops.add(fim, Image.frombytes('L', (length, length), bytes(r)), offset=-127)
            image.paste(fim, mask=Image.frombytes('L', (length, length), bytes(mask)))

        return image.crop((0, 0, self._size, self._size))
