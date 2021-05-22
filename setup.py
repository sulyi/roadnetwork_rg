from setuptools import setup

from src import demo_mapgen

name = 'demo_mapgen'
requires = ['Pillow']

setup(
    name=name,
    version=demo_mapgen.__version__,
    packages=[name],
    package_data={
        name: ['data/colourmap.palette'],
    },
    include_package_data=True,
    package_dir={'': 'src',
                 name: 'src/demo_mapgen'},
    url='',
    license='',
    author='Ákos Sülyi',
    author_email='sulyi.gbox@gmail.com',
    description='Generates height map with cities (and constructs a sensible road network)',
    install_requires=requires
)
