# -*- coding: utf-8 -*-
import os
import subprocess
import warnings

from setuptools import setup


def get_revision():
    # XXX: Use annotated (or signed) tags with names void of '-' to mark releases
    try:
        tag = subprocess.check_output([
            'git', 'describe', '--long', '--dirty=-dev0'
        ]).decode('utf-8')[:-1].split('-')
        rev = []
        if tag[1] != '0':
            rev.append(tag[1])
        if len(tag) > 3:
            rev.append(tag[3])
        return '.'.join(rev)
    except (subprocess.CalledProcessError, FileNotFoundError):
        warnings.warn("Revision number couldn't be determined")
        return None


def get_version(version_file):
    ver_ns = {}
    with open(version_file) as file:
        exec(file.read(), ver_ns)
    rev = get_revision()
    return ver_ns["__version__"] if not rev else ''.join((ver_ns["__version__"], 'rc', rev))


SRC = 'src'
NAME = 'roadnetwork_rg'
DATA = '%s.%s' % (NAME, 'data')

version = get_version(os.path.join(SRC, NAME, '_version.py'))

with open('README.md', 'r') as fp:
    readme = fp.read()

requires = ['Pillow', 'numpy']

# IDEA: use `setup.cfg`

setup(
    name=NAME,
    version=version,
    packages=[NAME, DATA],
    package_data={
        DATA: ['colourmap.palette'],  # just in case manifest is lost
    },
    include_package_data=True,
    package_dir={'': SRC,
                 NAME: '%s/%s' % (SRC, NAME),
                 DATA: '%s/%s/%s' % (SRC, NAME, 'data')},
    url='https://github.com/sulyi/roadnetwork_rg',
    license='GPLv3',
    platforms=['Linux'],
    author='Ákos Sülyi',
    author_email='sulyi.gbox@gmail.com',
    description='Generates height map and cities then constructs a sensible road network',
    long_description=readme,
    long_description_content_type='text/markdown',
    install_requires=requires
)
