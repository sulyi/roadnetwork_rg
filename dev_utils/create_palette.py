from itertools import chain, repeat

palette_gradient = []
with open('palette.txt', 'r') as txt:
    for line in txt:
        palette_gradient.append(line.strip())


def _interpolate(f_co, t_co, interval):
    det_co = [(t - f) / interval for f, t in zip(f_co, t_co)]
    for i in range(interval):
        yield tuple(round(f + det * i) for f, det in zip(f_co, det_co))


def get_palette(gradient):
    converted_colours = [(int(c[1:3], base=16), int(c[3:5], base=16), int(c[5:7], base=16)) for c in gradient]
    palette = list(chain(*chain(
        *(repeat(c, 8) for i in range(8) for c in _interpolate(converted_colours[i], converted_colours[i + 1], 4))
    )))
    return palette
