import bisect
from collections import Iterable
from typing import Union

from PIL.Image import Image

from .common import PointType


# TODO: check optimization
# pstats are in 'demo_mapgen_2021-05-24T16:27:21.prof' with
#     chunks = tuple((x, y) for x in range(0, 3) for y in range(-1, 1))
#     config = WorldConfig(128, .75, .55, 24, city_sizes=8)


class PixelPath:
    def __init__(self, cost: int, pixels: list[tuple[int, int]]) -> None:
        self.cost = cost
        self.pixels = pixels

    def __repr__(self):
        return "%s(%d, %s)" % (self.__class__.__name__, self.cost, self.pixels)


class Node:

    def __init__(self, x: int, y: int, distance: Union[float, None] = None) -> None:
        self._x = x
        self._y = y
        self.distance = distance
        self.parent: Union[tuple[int, int], None] = None
        self.open: bool = False

    def __repr__(self):
        return "%s(%d, %d, %s)" % (self.__class__.__name__, self._x, self._y,
                                   self.distance if self.distance is None else "%d" % self.distance)


def find_shortest_paths(graph: Image, source: PointType, targets: Iterable[PointType, ...]
                        ) -> dict[PointType, PixelPath]:
    source = source[:2]

    q = set(node[:2] for node in targets)
    new_nodes = [source]
    new_distances = [0.]

    sx, sy = graph.size
    nodes = {(x, y): Node(x, y) for x in range(sx) for y in range(sy)}
    nodes[source].distance = 0.
    nodes[source].open = True

    graph = list(graph.tobytes())

    d_cost = 2 ** .5

    while new_nodes:
        ux, uy = u = new_nodes.pop(0)
        new_distances.pop(0)
        nodes[u].open = False
        q.discard(u)

        u_value = graph[ux + sx * uy]
        for dx, dy, h_cost in ((-1, -1, d_cost), (0, -1, 1.), (1, -1, d_cost),
                               (-1, 0, 1.), (1, 0, 1.),
                               (-1, 1, d_cost), (0, 1, 1.), (1, 1, d_cost)):
            nx, ny = ux + dx, uy + dy
            if 0 <= nx < sx and 0 <= ny < sy:
                neighbour = (nx, ny)
                v_cost = abs(u_value - graph[nx + sx * ny])
                alt = nodes[u].distance + v_cost + h_cost
                if nodes[neighbour].distance is None or alt < nodes[neighbour].distance:
                    if nodes[neighbour].open:
                        index = new_nodes.index(neighbour)
                        new_nodes.pop(index)
                        new_distances.pop(index)
                    nodes[neighbour].distance = alt
                    nodes[neighbour].parent = u
                    nodes[neighbour].open = True
                    index = bisect.bisect(new_distances, alt)
                    new_distances.insert(index, alt)
                    new_nodes.insert(index, neighbour)
    if q:
        raise Exception("Couldn't be found a path to all nodes")

    paths = {target: PixelPath(nodes[target[:2]].distance, backtrack(nodes, target[:2])) for target in targets}
    return paths


def backtrack(predecessors: dict[tuple[int, int]: Node], current: tuple[int, int]) -> list[tuple[int, int], ...]:
    total_path = []
    while True:
        current = predecessors[current].parent
        if current is not None and predecessors[current].parent in predecessors:
            # None possible if multiple city in same node (one being the source)
            total_path.append(current)
            continue
        break
    total_path.reverse()
    return total_path
