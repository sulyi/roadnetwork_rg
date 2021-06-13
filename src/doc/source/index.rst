.. roadnetwork_rg documentation master file, created by
   sphinx-quickstart on Wed Jun  2 07:47:04 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to roadnetwork_rg's documentation!
==========================================

This python package implements a map generator that connects cities with a *sensible* road network.

The generated road network is close to a minimal spanning tree but allows loops, to achieve this:


* a height map is generated using diamond-square (a.k.a. random midpoint displacement) algorithm,
* a set of cities are generated using a point process with an adaptive intensity function,
* a complete graph of the shortest roads connecting the cities is created,
* a heuristic is applied to filter roads.

.. rubric:: height map

For persistent map generation coordinates are hashed, after seeding them instead of a pRNG stream.

.. rubric:: point process

To simulating pairwise interaction between generated points super Gaussian function with power of
1/3 is used as a potential field.

.. rubric:: shortest paths

In order to find every shortest path map is flooded by Dijkstra algorithm for each city.


.. rubric:: filtering heuristic

WIP, it is a greedy algorithm with exit condition of filtered graph being
continuous.

TODO
----


* intensity function of point process' expected value
* filter using value dispersion
* connect chunks

Contents:
---------

.. toctree::
   :maxdepth: 2

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
