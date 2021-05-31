from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Union, Sequence

from PIL import Image

from .common import PixelPath, PointType


@dataclass
class Node:
    # NOTE: 'x' and 'y' being properties would do more harm than good
    x: int
    y: int
    distance: float = float('inf')
    parent: Union[tuple[int, int], None] = None
    open: bool = False


class Pathfinder:
    __neighbours = ((-1, -1, 2 ** .5), (0, -1, 1.), (1, -1, 2 ** .5),
                    (-1, 0, 1.), (1, 0, 1.),
                    (-1, 1, 2 ** .5), (0, 1, 1.), (1, 1, 2 ** .5))

    def __init__(self, graph: Image.Image, targets: Sequence[PointType, ...]):
        self._width, self._height = graph.size
        self._graph = list(graph.tobytes())
        self._targets = targets

    # FIXME: check optimization

    def shortest_paths(self, source_index: int) -> dict[tuple[PointType, PointType], PixelPath]:
        nodes = {(x, y): Node(x, y) for x in range(self._width) for y in range(self._height)}
        nodes[self._targets[source_index][:2]].distance = 0.
        nodes[self._targets[source_index][:2]].open = True

        new_nodes = [Node(*self._targets[source_index][:2], distance=0.)]
        new_distances = [0.]

        while new_nodes:
            node = new_nodes.pop(0)
            new_distances.pop(0)
            node.open = False
            u_value = self._graph[node.x + self._width * node.y]

            for d_x, d_y, h_cost in self.__neighbours:
                n_x, n_y = node.x + d_x, node.y + d_y
                if 0 <= n_x < self._width and 0 <= n_y < self._height:
                    v_cost = abs(u_value - self._graph[n_x + self._width * n_y])
                    alt = node.distance + v_cost + h_cost
                    if alt < nodes[n_x, n_y].distance:
                        if nodes[n_x, n_y].open:
                            index = new_nodes.index(
                                nodes[n_x, n_y],
                                bisect.bisect_left(new_distances, nodes[n_x, n_y].distance)
                            )
                            new_nodes.pop(index)
                            new_distances.pop(index)
                        nodes[n_x, n_y].distance = alt
                        nodes[n_x, n_y].parent = node.x, node.y
                        nodes[n_x, n_y].open = True
                        index = bisect.bisect(new_distances, alt)
                        new_distances.insert(index, alt)
                        new_nodes.insert(index, nodes[n_x, n_y])

        self._check_result(nodes, source_index)

        return {(self._targets[source_index], target): PixelPath(nodes[target[:2]].distance,
                                                                 self.backtrack(nodes, target[:2]))
                for target in self._targets[source_index + 1:]}

    def _check_result(self, nodes, source_index):
        if {node[:2] for node in self._targets[source_index:]}.difference(nodes.keys()):
            # NOTE: shouldn't be reachable anyway
            raise ArithmeticError("Couldn't be found a path to each node")

    @staticmethod
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
