from setuptools import setup

from src import roadnetwork_rg

name = 'roadnetwork_rg'
requires = ['Pillow', 'numpy']

setup(
    name=name,
    version=roadnetwork_rg.__version__,
    packages=[name],
    package_data={
        name: ['data/colourmap.palette'],
    },
    include_package_data=True,
    package_dir={'': 'src',
                 name: 'src/roadnetwork_rg'},
    url='',
    license='',
    author='Ákos Sülyi',
    author_email='sulyi.gbox@gmail.com',
    description='Generates height map and cities then constructs a sensible road network',
    install_requires=requires
)
