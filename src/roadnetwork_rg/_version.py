# IDEA: generate version patch from git sha
# https://martin-thoma.com/python-package-versions/
__version__ = '0.1.0'
__version_info__ = tuple(int(i) for i in __version__.split('.') if i.isdigit())
