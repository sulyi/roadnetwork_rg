import hashlib
import random
from typing import Union, Any

SeedType = Union[None, int, str, bytes, bytearray]
PointType = tuple[int, int, int]


def safe_seed(seed: Any, bit_length: int) -> int:
    if not isinstance(bit_length, int):
        raise TypeError("Argument 'bit_length' should be integer number, not '%s'" % type(bit_length).__name__)
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
            raise TypeError("The only supported seed types are: None, int, str, bytes, and bytearray.")
    return seed
