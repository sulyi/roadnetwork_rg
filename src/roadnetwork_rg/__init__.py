__all__ = ("Datafile", "DatafileDecodeError", "DatafileEncodingError", "HeightMap", "PixelPath",
           "WorldChunk", "WorldChunkData", "WorldConfig", "WorldData", "WorldGenerator",
           "WorldRenderOptions", "colour_palette", "common", "datafile_version",
           "default_render_options", "default_world_config", "intensity", "pathfinder",
           "point_process")

# IDEA: generate version patch from git sha
__version__ = '0.1.0'
__version_info__ = tuple(int(i) for i in __version__.split('.') if i.isdigit())

from . import common, intensity, pathfinder, point_process
from .common import PixelPath, WorldChunkData, WorldConfig, WorldData, WorldRenderOptions
from .data import colour_palette
from .generator import WorldChunk, WorldGenerator, default_render_options, default_world_config
from .height_map import HeightMap
from .io import Datafile, DatafileDecodeError, DatafileEncodingError

datafile_version = Datafile.get_version()
