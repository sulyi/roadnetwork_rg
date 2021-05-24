import bisect
from collections import Iterable
from typing import Union

from PIL.Image import Image

from .common import PointType


class PixelPath:
    def __init__(self, cost: int, pixels: list[tuple[int, int]]) -> None:
        self.cost = cost
        self.pixels = pixels

    def __repr__(self):
        return "%s(%d, %s)" % (self.__class__.__name__, self.cost, self.pixels)

    def __eq__(self, other):
        return self is other or self.cost == other.cost and self.pixels == other.pixels


class Node:

    def __init__(self, x: int, y: int, distance: float = float('inf')) -> None:
        # XXX: 'x' and 'y' being properties would do more harm than good
        self.x = x
        self.y = y
        self.distance = distance
        self.parent: Union[tuple[int, int], None] = None
        self.open: bool = False

    def __repr__(self):
        return "%s(%d, %d, %s)" % (self.__class__.__name__, self.x, self.y,
                                   self.distance if self.distance is None else "%d" % self.distance)


def find_shortest_paths(graph: Image, source: PointType, targets: Iterable[PointType, ...]
                        ) -> dict[tuple[PointType, PointType], PixelPath]:
    sx, sy = graph.size
    nodes = {(x, y): Node(x, y) for x in range(sx) for y in range(sy)}
    nodes[source[:2]].distance = 0.
    nodes[source[:2]].open = True

    new_nodes = [Node(*source[:2], distance=0.)]
    new_distances = [0.]

    graph = list(graph.tobytes())

    d_cost = 2 ** .5

    while new_nodes:
        u = new_nodes.pop(0)
        new_distances.pop(0)
        u.open = False

        u_value = graph[u.x + sx * u.y]
        for dx, dy, h_cost in ((-1, -1, d_cost), (0, -1, 1.), (1, -1, d_cost),
                               (-1, 0, 1.), (1, 0, 1.),
                               (-1, 1, d_cost), (0, 1, 1.), (1, 1, d_cost)):
            nx, ny = u.x + dx, u.y + dy
            if 0 <= nx < sx and 0 <= ny < sy:
                neighbour = nodes[nx, ny]
                v_cost = abs(u_value - graph[nx + sx * ny])
                alt = u.distance + v_cost + h_cost
                if alt < neighbour.distance:
                    if neighbour.open:
                        index = new_nodes.index(neighbour, bisect.bisect_left(new_distances, neighbour.distance))
                        new_nodes.pop(index)
                        new_distances.pop(index)
                    neighbour.distance = alt
                    neighbour.parent = u.x, u.y
                    neighbour.open = True
                    index = bisect.bisect(new_distances, alt)
                    new_distances.insert(index, alt)
                    new_nodes.insert(index, neighbour)

    if {node[:2] for node in targets}.difference(nodes.keys()):
        raise Exception("Couldn't be found a path to all nodes")

    paths = {(source, target): PixelPath(nodes[target[:2]].distance,
                                         backtrack(nodes, target[:2])) for target in targets}
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
