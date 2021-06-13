# roadnetwork_rg #

This python package implements a map generator that connects cities with a *sensible* road network.

The generated road network is close to a minimal spanning tree but allows loops, to achieve this:

* a height map is generated using diamond-square (a.k.a. random midpoint displacement) algorithm,
* a set of cities are generated using a point process with an adaptive intensity function,
* a complete graph of the shortest roads connecting the cities is created,
* a heuristic is applied to filter roads.

## see also: ##

* [Documentation](http://sulyi.github.io/roadnetwork_rg)
