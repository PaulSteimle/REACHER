import sys

# Use built-in for Python 3.9+, fallback to backport for 3.8
if sys.version_info >= (3, 9):
    from importlib.resources import path
else:
    from importlib_resources import path

import os

def get_filter_path(filename):
    """
    Returns the file system path to a given .dat filter file.
    NOTE: Only works if the package is not zipped (e.g., not .egg/.zip).
    """
    from . import __name__ as pkg_name  # Equivalent to 'SIRP.profiles'
    with path(pkg_name, filename) as p:
        return os.fspath(p)

def list_filter_paths():
    """Return paths to all .dat files in the profiles package."""
    from . import __name__ as pkg_name
    if sys.version_info >= (3, 9):
        from importlib.resources import contents, is_resource
    else:
        from importlib_resources import contents, is_resource

    return [
        get_filter_path(name)
        for name in contents(pkg_name)
        if name.endswith(".dat") and is_resource(pkg_name, name)
    ]