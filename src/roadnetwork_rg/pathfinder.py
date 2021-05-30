from __future__ import annotations

import bisect
from typing import Iterable, Union

from PIL import Image

from .common import PixelPath, PointType


class Node:

    def __init__(self, x: int, y: int, *, distance: float = float('inf')) -> None:
        # XXX: 'x' and 'y' being properties would do more harm than good
        self.x = x
        self.y = y
        self.distance = distance
        self.parent: Union[tuple[int, int], None] = None
        self.open: bool = False

    def __repr__(self) -> str:
        return "%s(%d, %d, %s)" % (self.__class__.__name__, self.x, self.y,
                                   self.distance if self.distance is None else "%d" % self.distance)


def find_shortest_paths(graph: Image.Image, source: PointType, targets: Iterable[PointType, ...]
                        ) -> dict[tuple[PointType, PointType], PixelPath]:
    width, height = graph.size
    nodes = {(x, y): Node(x, y) for x in range(width) for y in range(height)}
    nodes[source[:2]].distance = 0.
    nodes[source[:2]].open = True

    new_nodes = [Node(*source[:2], distance=0.)]
    new_distances = [0.]

    graph = list(graph.tobytes())

    d = 2 ** .5

    while new_nodes:
        node = new_nodes.pop(0)
        new_distances.pop(0)
        node.open = False

        u_value = graph[node.x + width * node.y]
        for d_x, d_y, h_cost in ((-1, -1, d), (0, -1, 1.), (1, -1, d),
                                 (-1, 0, 1.), (1, 0, 1.),
                                 (-1, 1, d), (0, 1, 1.), (1, 1, d)):
            n_x, n_y = node.x + d_x, node.y + d_y
            if 0 <= n_x < width and 0 <= n_y < height:
                neighbour = nodes[n_x, n_y]
                v_cost = abs(u_value - graph[n_x + width * n_y])
                alt = node.distance + v_cost + h_cost
                if alt < neighbour.distance:
                    if neighbour.open:
                        start = bisect.bisect_left(new_distances, neighbour.distance)
                        index = new_nodes.index(neighbour, start)
                        new_nodes.pop(index)
                        new_distances.pop(index)
                    neighbour.distance = alt
                    neighbour.parent = node.x, node.y
                    neighbour.open = True
                    index = bisect.bisect(new_distances, alt)
                    new_distances.insert(index, alt)
                    new_nodes.insert(index, neighbour)

    if {node[:2] for node in targets}.difference(nodes.keys()):
        # XXX: shouldn't be reachable anyway
        raise ArithmeticError("Couldn't be found a path to each node")

    paths = {(source, target): PixelPath(nodes[target[:2]].distance,
                                         backtrack(nodes, target[:2])) for target in targets}
    return paths


def backtrack(predecessors: dict[tuple[int, int]: Node],
              current: tuple[int, int]) -> list[tuple[int, int], ...]:
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
