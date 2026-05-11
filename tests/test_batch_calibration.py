"""Tests for the Phase 4 calibration batch helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from REACHER.batch.calibration import (
    make_all_masterbackgrounds,
    make_all_masterdarks,
    make_all_masterflats,
    make_bad_pix_masks,
)
from REACHER.config import (
    CalibrationConfig,
    PathConfig,
    ProcessingConfig,
    ReacherConfig,
)
from REACHER.discovery.layout import DataLayout
from REACHER.exceptions import InvalidConfigError


def _make_config(synthetic_raw_layout, **overrides) -> ReacherConfig:
    paths = PathConfig(
        raw_input_dir=str(synthetic_raw_layout["raw"]),
        reduced_output_dir=str(synthetic_raw_layout["reduced"]),
    )
    processing = ProcessingConfig(make_plots=False, **overrides)
    return ReacherConfig(
        paths=paths, processing=processing,
        calibration=CalibrationConfig(),
    )


# ----------------------------------------------------------------------
# make_bad_pix_masks
# ----------------------------------------------------------------------
def test_make_bad_pix_masks_creates_mask_files(synthetic_raw_layout):
    config = _make_config(synthetic_raw_layout)
    layout = DataLayout(config.paths)

    pipeline = make_bad_pix_masks(config)
    assert len(pipeline.steps) == 1  # one obs_night

    bp_path = layout.bad_pix_mask_filepath("2020-01-01")
    clump_path = layout.clump_mask_filepath
    assert bp_path.exists()
    assert clump_path.exists()


def test_make_bad_pix_masks_resumability(synthetic_raw_layout):
    """Re-running with overwrite=False should skip existing outputs."""
    config = _make_config(synthetic_raw_layout, overwrite=False)
    layout = DataLayout(config.paths)

    make_bad_pix_masks(config)
    bp_path = layout.bad_pix_mask_filepath("2020-01-01")
    first_bytes = bp_path.read_bytes()

    # Re-run; the file should be untouched.
    make_bad_pix_masks(config)
    assert bp_path.read_bytes() == first_bytes


def test_make_bad_pix_masks_handles_empty_dark_df(synthetic_raw_layout):
    """An empty DARK DataFrame returns an empty pipeline (no error)."""
    import pandas as pd
    config = _make_config(synthetic_raw_layout)
    pipeline = make_bad_pix_masks(config, dark_df=pd.DataFrame())
    assert len(pipeline.steps) == 0


def test_make_bad_pix_masks_requires_config():
    with pytest.raises(InvalidConfigError):
        make_bad_pix_masks(None)


# ----------------------------------------------------------------------
# make_all_masterdarks
# ----------------------------------------------------------------------
def test_make_all_masterdarks_creates_per_exptime_files(synthetic_raw_layout):
    config = _make_config(synthetic_raw_layout)
    layout = DataLayout(config.paths)

    # bad pix mask is required by MasterDarkCreator if supplied; but our
    # batch helper passes None when missing, so this works either way.
    make_bad_pix_masks(config)
    pipeline = make_all_masterdarks(config)

    # We have two exptimes for the same obs_night => two steps.
    assert len(pipeline.steps) == 2

    out_t1 = layout.master_dark_filepath("2020-01-01", 1.0)
    out_t4 = layout.master_dark_filepath("2020-01-01", 4.0)
    assert out_t1.exists()
    assert out_t4.exists()


def test_make_all_masterdarks_resumability(synthetic_raw_layout):
    config = _make_config(synthetic_raw_layout, overwrite=False)
    layout = DataLayout(config.paths)

    make_bad_pix_masks(config)
    make_all_masterdarks(config)

    out_t1 = layout.master_dark_filepath("2020-01-01", 1.0)
    first_bytes = out_t1.read_bytes()

    make_all_masterdarks(config)
    assert out_t1.read_bytes() == first_bytes


# ----------------------------------------------------------------------
# make_all_masterflats
# ----------------------------------------------------------------------
def test_make_all_masterflats_creates_master_flat(synthetic_raw_layout):
    config = _make_config(synthetic_raw_layout)
    layout = DataLayout(config.paths)

    make_bad_pix_masks(config)
    make_all_masterdarks(config)
    pipeline = make_all_masterflats(config)

    # One filter per obs_night => one step.
    assert len(pipeline.steps) == 1

    mf = layout.master_flat_filepath("2020-01-01", "CI_H")
    assert mf.exists()


def test_make_all_masterflats_drops_rows_with_missing_dark(synthetic_raw_layout):
    """Without a master dark every flat row is dropped -> no steps."""
    config = _make_config(synthetic_raw_layout)
    layout = DataLayout(config.paths)
    layout.bad_pix_mask_folder.mkdir(parents=True, exist_ok=True)
    # Provide bad pix mask so the resolver can run; intentionally skip
    # master darks.
    (layout.bad_pix_mask_folder / "bad_pix_mask_2020-01-01.fits").write_bytes(b"")

    pipeline = make_all_masterflats(config)
    assert len(pipeline.steps) == 0


# ----------------------------------------------------------------------
# make_all_masterbackgrounds
# ----------------------------------------------------------------------
def test_make_all_masterbackgrounds_raises_not_implemented(synthetic_raw_layout):
    config = _make_config(synthetic_raw_layout)
    with pytest.raises(NotImplementedError):
        make_all_masterbackgrounds(config)
