from setuptools import setup, find_packages
import REACHER

setup(
    name='REACHER',
    version= REACHER.__version__,
    description='Pre-processing pipeline for SPHERE IRDIS Dual Band Imaging',
    author='Paul Steimle',
    author_email='pasteimle@mpia.de',
    license='MIT license',
    packages= find_packages(),
    package_data={"REACHER.profiles": ["*.dat"]},  # Include all .dat files in profiles
    install_requires=[]
)