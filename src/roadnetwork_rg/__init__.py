"""Generates a road network

Map is broken up to chunks that are generated in a persistent manner.
Three steps are taken while generating a map of a road network:

* a height map is generated using diamond-square algorithm, a.k.a. random midpoint displacement
    algorithm,
* a set of cities are generated using a spatial point process,
* shortest roads between the cities are found and filtered based on a heuristic.

Modules include:

    :mod:`roadnetwork_rg.intensity`
        Various implementation of :class:`.point_process.IntensityFunction` and other classes
        utilized by them.

    :mod:`roadnetwork_rg.pathfinder`
        A pathfinder algorithm that floods a graph from a source and find shortest paths to all of a
        set of targets.

    :mod:`roadnetwork_rg.point_process`
        A couple of :class:`~collection.abc.Iterator` to generate point processes and an abstract
        :class:`~.point_process.IntensityFunction` providing an interface used by them.

"""

# TODO: check doc

__all__ = ("Datafile", "DatafileDecodeError", "DatafileEncodingError", "HeightMap",
           "HeightMapConfig", "PixelPath", "PointType", "SeedType", "WorldChunk", "WorldChunkData",
           "WorldConfig", "WorldData", "WorldGenerator", "WorldRenderOptions", "colour_palette",
           "default_render_options", "default_world_config", "intensity", "get_safe_seed",
           "pathfinder", "point_process")

from . import intensity, pathfinder, point_process, _version
from .common import (HeightMapConfig, PixelPath, PointType, SeedType, WorldChunkData, WorldConfig,
                     WorldData, WorldRenderOptions, get_safe_seed)
from .data import colour_palette
from .datafile import Datafile, DatafileDecodeError, DatafileEncodingError
from .height_map import HeightMap
from .world import WorldChunk, WorldGenerator, default_render_options, default_world_config

__datafile_version__ = Datafile.get_version()
"""Version of :class:`.Datafile` file format"""
__version__ = _version.__version__
"""Version of package"""
__version_info__ = tuple(int(i) for i in __version__.split('.') if i.isdigit())
