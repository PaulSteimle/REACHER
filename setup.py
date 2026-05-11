"""Setuptools entry point for REACHER."""
from pathlib import Path
import re

from setuptools import find_packages, setup


_HERE = Path(__file__).parent


def _read_version() -> str:
    """Pull __version__ from REACHER/__init__.py without importing it."""
    init = (_HERE / "REACHER" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init, re.M)
    if not match:
        raise RuntimeError("Cannot find __version__ in REACHER/__init__.py")
    return match.group(1)


def _read_long_description() -> str:
    readme = _HERE / "README.md"
    if readme.exists():
        return readme.read_text(encoding="utf-8")
    return ""


setup(
    name="REACHER",
    version=_read_version(),
    description="Raw image Extraction And Calibration for High-contrast Exoplanet Research — VLT/SPHERE-IRDIS pre-processing pipeline",
    long_description=_read_long_description(),
    long_description_content_type="text/markdown",
    author="Paul Steimle",
    author_email="pasteimle@mpia.de",
    url="https://github.com/PaulSteimle/REACHER",
    license="MIT",
    packages=find_packages(exclude=("tests", "tests.*")),
    package_data={
        # SPHERE/IRDIS filter transmission curves consumed by utils/filters.py.
        "REACHER.irdis_filters": ["*.dat"],
    },
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24,<2",
        "scipy>=1.10",
        "pandas>=2.0",
        "astropy>=5.2",
        "vip-hci==1.6.6",
        "photutils>=1.8",
        "matplotlib>=3.7",
        # Used by vip-hci for anamorphic resampling (imlib='opencv').
        "opencv-python>=4.5",
        # Speeds up cosmic-ray median ops; soft-imported with a numpy
        # fallback in REACHER/utils/cosmics.py, but real installs should
        # have it for performance.
        "bottleneck>=1.3",
    ],
    extras_require={
        "dev": ["pytest>=7"],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
)
