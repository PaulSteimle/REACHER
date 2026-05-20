"""Dataset discovery layer (Phase 4).

Replaces the retired ``Parameters`` / ``FILEHANDLER`` /
``make_*_DF`` / ``find_calib_files`` chain on top of the new
:class:`ReacherConfig` dataclass interface.

Public API:

* :class:`StarTarget`           -- per-target value object.
* :class:`DataLayout`           -- on-disk path resolver.
* :func:`list_directory`        -- macOS-aware directory listing.
* :func:`make_dark_DF`,
  :func:`make_flat_DF`,
  :func:`make_background_DF`,
  :func:`make_star_DF`          -- DataFrame builders.
* :func:`list_target_stars`,
  :func:`iter_star_targets`     -- target iteration helpers.
* :func:`resolve_star_calibration` -- per-row calibration matcher.
"""
from .calib_resolver import resolve_star_calibration
from .dataframes import (
    iter_star_targets,
    list_target_stars,
    make_background_DF,
    make_dark_DF,
    make_flat_DF,
    make_star_DF,
    stars_from_ref_cubes,
)
from .layout import DataLayout, list_directory
from .target import StarTarget

__all__ = [
    "DataLayout",
    "StarTarget",
    "iter_star_targets",
    "list_directory",
    "list_target_stars",
    "make_background_DF",
    "make_dark_DF",
    "make_flat_DF",
    "make_star_DF",
    "resolve_star_calibration",
    "stars_from_ref_cubes",
]
