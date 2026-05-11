"""Batch helper functions (Phase 4 user-facing layer).

These helpers are thin wrappers that:

1. discover the on-disk layout via :mod:`REACHER.discovery`,
2. instantiate the relevant :class:`PipelineStep` subclasses,
3. wire them into a :class:`Pipeline`,
4. call ``pipeline.run()``.

They are the Phase 4 replacement for the legacy
``master_routines`` / ``misc`` user-facing functions; their names
match the legacy verbatim so existing run scripts can be ported with
minimal lift.
"""
from .calibration import (
    make_all_masterbackgrounds,
    make_all_masterdarks,
    make_all_masterflats,
    make_bad_pix_masks,
)
from .cubes import centering_and_bad_pix_correction
from .extraction import pre_calibration_and_extraction
from .inspection import (
    cleanup_all_folders,
    grab_all_files,
    return_csv_files,
)
from ..discovery.layout import list_directory


__all__ = [
    "centering_and_bad_pix_correction",
    "cleanup_all_folders",
    "grab_all_files",
    "list_directory",
    "make_all_masterbackgrounds",
    "make_all_masterdarks",
    "make_all_masterflats",
    "make_bad_pix_masks",
    "pre_calibration_and_extraction",
    "return_csv_files",
]
