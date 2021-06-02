__all__ = ("Datafile", "DatafileDecodeError", "DatafileEncodingError", "HeightMap",
           "HeightMapConfig", "PixelPath", "WorldChunk", "WorldChunkData", "WorldConfig",
           "WorldData", "WorldGenerator", "WorldRenderOptions", "colour_palette", "common",
           "default_render_options", "default_world_config", "intensity", "pathfinder",
           "point_process")

# IDEA: generate version patch from git sha
# https://martin-thoma.com/python-package-versions/
__version__ = '0.1.0'
__version_info__ = tuple(int(i) for i in __version__.split('.') if i.isdigit())

from . import common, intensity, pathfinder, point_process
from .common import (HeightMapConfig, PixelPath, WorldChunkData, WorldConfig, WorldData,
                     WorldRenderOptions)
from .data import colour_palette
from .datafile import Datafile, DatafileDecodeError, DatafileEncodingError
from .height_map import HeightMap
from .world import WorldChunk, WorldGenerator, default_render_options, default_world_config

__datafile_version = Datafile.get_version()
