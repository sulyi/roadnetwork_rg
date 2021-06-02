# -*- coding: utf-8 -*-
import pkg_resources
from setuptools import setup

NAME = 'roadnetwork_rg'
SRC = 'src'
version = pkg_resources.get_distribution(NAME).version

requires = ['Pillow', 'numpy']

# FIXME: use `setup.cfg`

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
