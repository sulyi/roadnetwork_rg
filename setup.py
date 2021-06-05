# -*- coding: utf-8 -*-
import subprocess
import warnings

from setuptools import setup

from src.roadnetwork_rg import __version__ as version


def get_revision():
    try:
        tags = (subprocess.check_output(['git', 'describe', '--dirty=-dev'])
                ).decode('utf-8')[:-1].split('-')
        return '-'.join((*tags[1:2], *tags[3:]))
    except (subprocess.CalledProcessError, FileNotFoundError):
        warnings.warn("Revision number couldn't be determined")
        return None


NAME = 'roadnetwork_rg'
SRC = 'src'

rev = get_revision()
version = version if not rev else ''.join((version, 'rc', rev))

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
