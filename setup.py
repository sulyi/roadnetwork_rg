# -*- coding: utf-8 -*-
import os
import subprocess
import warnings

from setuptools import setup


def get_revision():
    # NOTE: Use annotated (or signed) tags with names void of '-' to mark releases
    try:
        tags = (subprocess.check_output(['git', 'describe', '--dirty=-dev0'])
                ).decode('utf-8')[:-1].split('-')
        return '.'.join((*tags[1:2], *tags[3:]))
    except (subprocess.CalledProcessError, FileNotFoundError):
        warnings.warn("Revision number couldn't be determined")
        return None


def get_version(version_file):
    ver_ns = {}
    with open(version_file) as f:
        exec(f.read(), ver_ns)
    rev = get_revision()
    return ver_ns["__version__"] if not rev else ''.join((ver_ns["__version__"], 'rc', rev))


NAME = 'roadnetwork_rg'
SRC = 'src'

version = get_version(os.path.join(SRC, NAME, '_version.py'))

requires = ['Pillow', 'numpy']

# IDEA: use `setup.cfg`

# TODO: add following options
# license
# github url
# platform
# ...

setup(
    name=NAME,
    version=version,
    packages=[NAME],
    package_data={
        NAME: ['data/colourmap.palette'],
    },
    include_package_data=True,
    package_dir={'': SRC,
                 NAME: '%s/%s' % (SRC, NAME)},
    url='http://localhost',
    license='',
    author='Ákos Sülyi',
    author_email='sulyi.gbox@gmail.com',
    description='Generates height map and cities then constructs a sensible road network',
    install_requires=requires
)
