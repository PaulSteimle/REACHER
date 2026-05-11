"""Batch helpers for calibration master products.

Each helper builds a :class:`Pipeline` from one or more
:class:`PipelineStep` instances and runs it. This is the Phase 4
replacement for the legacy
``master_routines.make_bad_pix_masks`` /
``master_routines.make_all_masterdarks`` /
``master_routines.make_all_masterflats`` chain.

The helpers are deliberately thin: dataset discovery (via
:mod:`REACHER.discovery`) -> step instantiation -> :class:`Pipeline`
execution. They never mutate the configuration.
"""
from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from ..config import ReacherConfig
from ..exceptions import InvalidConfigError
from ..pipeline import Pipeline
from ..steps.bad_pixel import BadPixelMaskCreator
from ..steps.dark import MasterDarkCreator
from ..steps.flat import MasterFlatCreator
from ..discovery.dataframes import (
    make_dark_DF,
    make_flat_DF,
)
from ..discovery.layout import DataLayout
from ..discovery.calib_resolver import resolve_star_calibration


logger = logging.getLogger(__name__)


def _check_config(config: ReacherConfig) -> None:
    if config is None:
        raise InvalidConfigError("config must not be None")
    if config.paths is None:
        raise InvalidConfigError("config.paths must not be None")


def make_bad_pix_masks(
    config: ReacherConfig,
    *,
    dark_df: Optional[pd.DataFrame] = None,
    configure_logging: bool = False,
) -> Pipeline:
    """Create one bad-pixel mask per ``obs_night`` plus the IRDIS clump mask.

    Parameters
    ----------
    config
        :class:`ReacherConfig` carrying ``paths.raw_input_dir`` and
        ``paths.reduced_output_dir``.
    dark_df
        Optional pre-built DARK DataFrame (e.g. cached from a previous
        call). When ``None`` (default) a fresh DataFrame is built via
        :func:`REACHER.discovery.make_dark_DF`.
    configure_logging
        Forwarded to :class:`Pipeline`. Off by default so the caller
        controls log-file initialisation (mirrors the legacy script
        which calls ``setup_logging`` once at module import time).

    Returns
    -------
    Pipeline
        The executed pipeline (steps are recorded in
        ``pipeline.steps``).

    Notes
    -----
    The clump mask is generated alongside the **first** night's
    ``BadPixelMaskCreator``: that step writes both
    ``bad_pix_mask_<obs_night>.fits`` and the shared
    ``IRDIS_clump_mask.fits``. Subsequent nights re-use the same clump
    mask path; resumability ensures it is only computed once.
    """
    _check_config(config)
    layout = DataLayout(config.paths)
    layout.bad_pix_mask_folder.mkdir(parents=True, exist_ok=True)

    if dark_df is None:
        dark_df = make_dark_DF(
            config.paths.raw_input_dir, instrument=config.instrument,
        )
    if dark_df.empty:
        logger.warning(
            "make_bad_pix_masks: DARK_DF is empty (no DARK files found "
            "under %s); nothing to do",
            config.paths.raw_input_dir,
        )
        return Pipeline(config, configure_logging=configure_logging)

    pipeline = Pipeline(config, configure_logging=configure_logging)

    obs_nights = sorted(dark_df["obs_night"].unique())
    clump_mask_path = layout.clump_mask_filepath

    for obs_night in obs_nights:
        dark_files = dark_df[dark_df["obs_night"] == obs_night]["filepath"].tolist()
        bad_pix_mask_path = layout.bad_pix_mask_filepath(obs_night)
        pipeline.add_step(BadPixelMaskCreator(
            config=config,
            dark_files=dark_files,
            bad_pix_mask_path=bad_pix_mask_path,
            clump_mask_path=clump_mask_path,
        ))

    pipeline.run()
    return pipeline


def make_all_masterdarks(
    config: ReacherConfig,
    *,
    dark_df: Optional[pd.DataFrame] = None,
    configure_logging: bool = False,
) -> Pipeline:
    """Create one master dark per ``(obs_night, exp_time)``.

    Parameters
    ----------
    config
        :class:`ReacherConfig`.
    dark_df
        Optional pre-built DARK DataFrame. Built via
        :func:`make_dark_DF` when ``None``.
    configure_logging
        Forwarded to :class:`Pipeline`.

    Returns
    -------
    Pipeline
        The executed pipeline.

    Notes
    -----
    Each :class:`MasterDarkCreator` instance is given the
    matching bad-pix mask + clump mask paths so that the dark is
    bad-pixel corrected. If a bad-pix mask is missing
    :class:`MasterDarkCreator._check_inputs` raises
    ``StepExecutionError`` -- run :func:`make_bad_pix_masks` first.
    """
    _check_config(config)
    layout = DataLayout(config.paths)
    layout.master_dark_folder.mkdir(parents=True, exist_ok=True)

    if dark_df is None:
        dark_df = make_dark_DF(
            config.paths.raw_input_dir, instrument=config.instrument,
        )
    if dark_df.empty:
        logger.warning(
            "make_all_masterdarks: DARK_DF is empty; nothing to do"
        )
        return Pipeline(config, configure_logging=configure_logging)

    pipeline = Pipeline(config, configure_logging=configure_logging)
    clump_mask_path = layout.clump_mask_filepath

    obs_nights = sorted(dark_df["obs_night"].unique())
    for obs_night in obs_nights:
        bad_pix_mask_path = layout.bad_pix_mask_filepath(obs_night)
        if not bad_pix_mask_path.exists():
            logger.warning(
                "make_all_masterdarks: bad pix mask %s missing; "
                "MasterDarkCreator will fail. Run make_bad_pix_masks "
                "first.",
                bad_pix_mask_path.name,
            )
            bad_pix_arg = None
        else:
            bad_pix_arg = bad_pix_mask_path

        clump_arg = clump_mask_path if clump_mask_path.exists() else None

        night_df = dark_df[dark_df["obs_night"] == obs_night]
        for exp_time in sorted(night_df["exp_time"].unique()):
            files = night_df[night_df["exp_time"] == exp_time][
                "filepath"
            ].tolist()
            master_dark_path = layout.master_dark_filepath(obs_night, exp_time)
            pipeline.add_step(MasterDarkCreator(
                config=config,
                dark_files=files,
                master_dark_path=master_dark_path,
                bad_pix_mask_path=bad_pix_arg,
                clump_mask_path=clump_arg,
            ))

    pipeline.run()
    return pipeline


def make_all_masterflats(
    config: ReacherConfig,
    *,
    flat_df: Optional[pd.DataFrame] = None,
    configure_logging: bool = False,
) -> Pipeline:
    """Create one master flat per ``(obs_night, filter_name)``.

    Parameters
    ----------
    config
        :class:`ReacherConfig`.
    flat_df
        Optional pre-built FLAT DataFrame.
    configure_logging
        Forwarded to :class:`Pipeline`.

    Returns
    -------
    Pipeline
        The executed pipeline.

    Notes
    -----
    The legacy routine resolved per-flat master darks through
    :func:`find_calib_files` with ``calib_type='master_flat'``. Phase 4
    keeps that exact behaviour via
    :func:`REACHER.discovery.resolve_star_calibration`.
    Flat rows whose master dark is missing are silently dropped
    (matches the legacy ``DF[DF['master_dark'] != 'None']`` filter).
    """
    _check_config(config)
    layout = DataLayout(config.paths)
    layout.master_flat_folder.mkdir(parents=True, exist_ok=True)

    if flat_df is None:
        flat_df = make_flat_DF(
            config.paths.raw_input_dir, instrument=config.instrument,
        )
    if flat_df.empty:
        logger.warning(
            "make_all_masterflats: FLAT_DF is empty; nothing to do"
        )
        return Pipeline(config, configure_logging=configure_logging)

    flat_df = resolve_star_calibration(
        flat_df, config, layout=layout, calib_type="master_flat"
    )
    flat_df = flat_df[flat_df["master_dark"] != "None"]
    if flat_df.empty:
        logger.warning(
            "make_all_masterflats: every flat row was dropped (no "
            "matching master darks). Run make_all_masterdarks first."
        )
        return Pipeline(config, configure_logging=configure_logging)

    pipeline = Pipeline(config, configure_logging=configure_logging)
    clump_mask_path = layout.clump_mask_filepath
    clump_arg = clump_mask_path if clump_mask_path.exists() else None

    for obs_night in sorted(flat_df["obs_night"].unique()):
        night_df = flat_df[flat_df["obs_night"] == obs_night]
        for filter_name in sorted(night_df["filter"].unique()):
            sub = night_df[night_df["filter"] == filter_name]
            unique_masks = sub["bad_pix_mask"].unique()
            if len(unique_masks) > 1:
                raise InvalidConfigError(
                    f"more than one bad pixel mask found for obs night "
                    f"{obs_night} and filter {filter_name}; "
                    f"got {list(unique_masks)}"
                )
            bad_pix_arg = unique_masks[0] if unique_masks[0] else None

            master_flat_path = layout.master_flat_filepath(
                obs_night, filter_name
            )
            pipeline.add_step(MasterFlatCreator(
                config=config,
                flat_files=sub["filepath"].tolist(),
                master_dark_files=sub["master_dark"].tolist(),
                master_flat_path=master_flat_path,
                bad_pix_mask_path=bad_pix_arg,
                clump_mask_path=clump_arg,
            ))

    pipeline.run()
    return pipeline


def make_all_masterbackgrounds(
    config: ReacherConfig,
    *,
    flat_df: Optional[pd.DataFrame] = None,
    configure_logging: bool = False,
) -> Pipeline:
    """Master-background creation (NOT IMPLEMENTED).

    The legacy ``make_all_masterbackgrounds`` shipped without a working
    body -- the wrapper it called raised a silent ``NotImplementedError``
    that the caller was expected to swallow. Architecture Rule 7 forbids
    that pattern.

    Raises
    ------
    NotImplementedError
        Always. Use :class:`MasterSkyCreator` for sky / background
        subtraction instead.
    """
    _check_config(config)
    raise NotImplementedError(
        "make_all_masterbackgrounds is not implemented. The legacy "
        "wrapper that this helper would have called was incomplete in "
        "the original codebase (see make_master_bkg_wrapper). For sky "
        "subtraction use the MasterSkyCreator step (Phase 2 Step 13)."
    )
