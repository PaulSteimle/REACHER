from setuptools import setup, find_packages

setup(
    name='SIRP',
    version='0.1.0',
    description='Preprocessing pipeline for SPHERE IRDIS Dual Band Imaging',
    author='Paul Steimle',
    author_email='pasteimle@mpia.de',
    license='MIT license',
    packages= find_packages(),
    package_data={"SIRP.profiles": ["*.dat"]},  # Include all .dat files in profiles
    install_requires=[]
)