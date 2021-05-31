"""Binary data for `roadnetwork_rg`"""

__all__ = ["colour_palette"]

from importlib import resources

colour_palette = resources.open_binary(__name__, 'colourmap.palette').read()
"""colour palette for `HeightMap` with levels"""
