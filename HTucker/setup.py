from setuptools import setup

setup(
    name='HTucker',
    version='0.1',
    description='An implementation of the hierarchical Tuckerformat for tensors',
    author='Benedikt Schroeter',
    author_email='b.schroeter@em.uni-frankfurt.de',
    packages=['HTucker'],
    install_requires=[
        'numpy',
        'torch',
    ],
)