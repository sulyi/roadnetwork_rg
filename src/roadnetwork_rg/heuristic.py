""""""
# FIXME: add doc

from __future__ import annotations

from typing import List, Dict, Tuple, Set

from .common import PointType, PixelPath


# TODO: update __init__, __init__.__all__, doc autosummary, toctree submodules


def greedy(targets: List[PointType, ...], paths: Dict[Tuple[PointType, PointType], PixelPath]
           ) -> Set[Tuple[PointType, PointType], ...]:
    """Selects cheapest road until all target are connected.

    :param targets: It is a list of points to be connected.
    :type targets: :class:`list` [:data:`.PointType`, ...]
    :param paths: It is a dictionary of available paths.
    :type paths: :class:`dict` [:class:`tuple` [:data:`.PointType`, :data:`.PointType`],
        :class:`.PixelPath`]
    :return: It is a set of keys corresponding to selected paths.
    :rtype: :class:`set` [:class:`tuple` [:data:`.PointType`, :data:`.PointType`], ...]
    """
    sorted_paths = sorted(paths, key=lambda path: paths[path].cost, reverse=True)
    lookup = {key: 0 for key in targets}
    available = 1
    q = {0: targets.copy()}
    result = set()
    if sorted_paths:
        while True:
            cheapest = sorted_paths.pop()
            result.add(cheapest)
            if lookup[cheapest[0]] == 0 and lookup[cheapest[1]] == 0:
                q[available] = [*cheapest]
                q[0].remove(cheapest[0])
                q[0].remove(cheapest[1])
                lookup[cheapest[0]] = available
                lookup[cheapest[1]] = available
                available += 1
                if not q[0]:
                    q.pop(0)
            elif lookup[cheapest[0]] == 0 or lookup[cheapest[1]] == 0:
                source, target = cheapest if lookup[cheapest[0]] == 0 else reversed(cheapest)
                q[0].remove(source)
                q[lookup[target]].append(source)
                lookup[source] = lookup[target]
                if not q[0]:
                    q.pop(0)
            elif lookup[cheapest[0]] != lookup[cheapest[1]]:
                target, source = sorted(cheapest, key=lookup.get)
                q_index = lookup[source]
                q[lookup[target]].extend(q[lookup[source]])
                for city in q[lookup[source]]:
                    lookup[city] = lookup[target]
                q.pop(q_index)

            if len(q) == 1:
                break
    return result
