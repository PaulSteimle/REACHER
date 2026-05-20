__version__ = "0.6.0-dev"

"""REACHER -- VLT/SPHERE IRDIS pre-processing pipeline.

Public API after Phase 4:

* :class:`Pipeline`            -- top-level orchestration class.
* :class:`ReacherConfig`       -- composed configuration dataclass.
* :class:`PipelineStep`        -- abstract base for all steps.
* The nine concrete step classes under ``REACHER.steps``.
* :func:`setup_logging`        -- per-run log-file initializer.

Phase 4 user-facing layer:

* :class:`StarTarget`               -- per-target value object.
* :class:`DataLayout`               -- on-disk path resolver.
* Discovery helpers: :func:`make_dark_DF`, :func:`make_flat_DF`,
  :func:`make_background_DF`, :func:`make_star_DF`,
  :func:`list_target_stars`, :func:`iter_star_targets`,
  :func:`resolve_star_calibration`, :func:`list_directory`.
* Batch helpers (matching the legacy script API):
  :func:`make_bad_pix_masks`, :func:`make_all_masterdarks`,
  :func:`make_all_masterflats`, :func:`make_all_masterbackgrounds`
  (raises :class:`NotImplementedError`),
  :func:`pre_calibration_and_extraction`,
  :func:`centering_and_bad_pix_correction`,
  :func:`grab_all_files`, :func:`return_csv_files`,
  :func:`cleanup_all_folders`.
"""
from .config import (
    CalibrationConfig,
    InstrumentConfig,
    PathConfig,
    ProcessingConfig,
    ReacherConfig,
)
from .exceptions import (
    CalibrationFileNotFoundError,
    DataFrameError,
    FITSReadError,
    FITSWriteError,
    HeaderKeyError,
    InvalidConfigError,
    ReacherError,
    ShapeMismatchError,
    StepExecutionError,
)
from .pipeline import ParallelStepGroup, Pipeline
from .steps import (
    BadPixelMaskCreator,
    CenteringPreparer,
    ChannelSplitter,
    CubeAssembler,
    FileCalibrator,
    FluxCalibrator,
    MasterDarkCreator,
    MasterFlatCreator,
    MasterSkyCreator,
    PipelineStep,
)
from .utils.banner import print_banner
from .utils.download import move_files_from_download_dir
from .utils.logging import setup_logging

# Phase 4: dataset discovery layer + batch helpers.
from .discovery import (
    DataLayout,
    StarTarget,
    iter_star_targets,
    list_directory,
    list_target_stars,
    make_background_DF,
    make_dark_DF,
    make_flat_DF,
    make_star_DF,
    resolve_star_calibration,
    stars_from_ref_cubes,
)
from .batch import (
    centering_and_bad_pix_correction,
    cleanup_all_folders,
    grab_all_files,
    make_all_masterbackgrounds,
    make_all_masterdarks,
    make_all_masterflats,
    make_bad_pix_masks,
    pre_calibration_and_extraction,
    return_csv_files,
)

__all__ = [
    # Steps
    "BadPixelMaskCreator",
    "CenteringPreparer",
    "ChannelSplitter",
    "CubeAssembler",
    "FileCalibrator",
    "FluxCalibrator",
    "MasterDarkCreator",
    "MasterFlatCreator",
    "MasterSkyCreator",
    "PipelineStep",
    # Pipeline + config
    "ParallelStepGroup",
    "Pipeline",
    "ReacherConfig",
    "CalibrationConfig",
    "InstrumentConfig",
    "PathConfig",
    "ProcessingConfig",
    "print_banner",
    "setup_logging",
    # Exceptions
    "CalibrationFileNotFoundError",
    "DataFrameError",
    "FITSReadError",
    "FITSWriteError",
    "HeaderKeyError",
    "InvalidConfigError",
    "ReacherError",
    "ShapeMismatchError",
    "StepExecutionError",
    # Phase 4 -- discovery
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
    # Phase 4 -- batch
    "centering_and_bad_pix_correction",
    "cleanup_all_folders",
    "grab_all_files",
    "make_all_masterbackgrounds",
    "make_all_masterdarks",
    "make_all_masterflats",
    "make_bad_pix_masks",
    "pre_calibration_and_extraction",
    "return_csv_files",
    # Pre-pipeline utility
    "move_files_from_download_dir",
]
