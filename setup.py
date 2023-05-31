import setuptools
from pathlib import Path

setuptools.setup(
    name='gym_franka',
    author="Realeboga Dikole",
    version='0.0.1',
    install_requires=['gym', 'pybullet', 'numpy'],  # And any other dependencies foo needs    
)