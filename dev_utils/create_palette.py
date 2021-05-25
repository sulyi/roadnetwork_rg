# TODO: document dev_utils.create_palette

import os

from itertools import chain, repeat

filename = os.path.join(os.path.dirname(__file__), 'palette.txt')

with open(filename, 'r') as txt:
    palette_gradient = [line.strip() for line in txt]


def _interpolate(f_co, t_co, interval):
    det_co = [(t - f) / interval for f, t in zip(f_co, t_co)]
    for i in range(interval):
        yield tuple(round(f + det * i) for f, det in zip(f_co, det_co))


def get_palette(gradient):
    rgb_colours = [
        (int(c[1:3], base=16),
         int(c[3:5], base=16),
         int(c[5:7], base=16)) for c in gradient
    ]
    palette = bytes(
        chain(*chain(
            # TODO: Add possibility to create smooth palette
            *(repeat(c, 8)
              for i in range(8) for c in _interpolate(rgb_colours[i], rgb_colours[i + 1], 4)
              )
        ))
    )
    return palette
