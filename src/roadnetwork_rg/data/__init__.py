from importlib import resources

colour_palette = resources.open_binary(__name__, 'colourmap.palette').read()
