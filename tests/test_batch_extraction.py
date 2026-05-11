"""Tests for the Phase 4 pre_calibration_and_extraction batch helper."""
from __future__ import annotations

from pathlib import Path

import pytest
from astropy.io import fits

from REACHER.batch.calibration import (
    make_all_masterdarks,
    make_all_masterflats,
    make_bad_pix_masks,
)
from REACHER.batch.extraction import pre_calibration_and_extraction
from REACHER.pipeline import ParallelStepGroup
from REACHER.steps.calibrate import FileCalibrator
from REACHER.steps.centering_prep import CenteringPreparer
from REACHER.steps.splitting import ChannelSplitter
from REACHER.config import (
    CalibrationConfig,
    PathConfig,
    ProcessingConfig,
    ReacherConfig,
)
from REACHER.discovery.layout import DataLayout
from REACHER.discovery.target import StarTarget
from REACHER.exceptions import InvalidConfigError, StepExecutionError


def _make_config(synthetic_raw_layout, **overrides) -> ReacherConfig:
    paths = PathConfig(
        raw_input_dir=str(synthetic_raw_layout["raw"]),
        reduced_output_dir=str(synthetic_raw_layout["reduced"]),
    )
    processing = ProcessingConfig(make_plots=False, **overrides)
    return ReacherConfig(
        paths=paths,
        processing=processing,
        calibration=CalibrationConfig(),
    )


def _setup_calibration_files(config):
    """Run the three calibration batch helpers, in order."""
    make_bad_pix_masks(config)
    make_all_masterdarks(config)
    make_all_masterflats(config)


def _target_for(layout_dict) -> StarTarget:
    return StarTarget(
        star_name=layout_dict["star"],
        obs_night=layout_dict["obs_night"],
        filter_name=layout_dict["filter_name"],
    )


# ----------------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------------
def test_pre_calibration_and_extraction_happy_path(synthetic_raw_layout):
    config = _make_config(synthetic_raw_layout)
    _setup_calibration_files(config)

    target = _target_for(synthetic_raw_layout)
    pipeline = pre_calibration_and_extraction(config, target)

    layout = DataLayout(config.paths)
    stage1 = layout.stage1_output_folder(
        target.star_name, target.obs_night, target.filter_name
    )
    stage2 = layout.stage2_output_folder(
        target.star_name, target.obs_night, target.filter_name
    )

    # 4 raw files -> 4 calibrated + 4 LEFT splits + 4 RIGHT splits.
    assert (stage1 / "CALIB_SPHER_SCIENCE_001.fits").exists()
    assert (stage1 / "CALIB_SPHER_FLUX_001.fits").exists()
    assert (stage1 / "CALIB_SPHER_CENTER_001.fits").exists()
    assert (stage1 / "CALIB_SPHER_SKY_001.fits").exists()
    assert (stage2 / "LEFT_SPHER_SCIENCE_001.fits").exists()
    assert (stage2 / "RIGHT_SPHER_SCIENCE_001.fits").exists()
    assert (stage2 / "LEFT_SPHER_CENTER_001.fits").exists()
    assert (stage2 / "RIGHT_SPHER_CENTER_001.fits").exists()
    # CENTER files trigger the centers.csv build.
    assert (stage2 / "centers.csv").exists()

    # Pipeline wires up two parallel groups (calibrate + split) plus
    # one CenteringPreparer = 3 outer steps. Each parallel group holds
    # one inner step per raw file.
    assert len(pipeline.steps) == 3
    calibrate_group, split_group, centering = pipeline.steps
    assert isinstance(calibrate_group, ParallelStepGroup)
    assert isinstance(split_group, ParallelStepGroup)
    assert isinstance(centering, CenteringPreparer)
    assert all(
        isinstance(s, FileCalibrator)
        for s in calibrate_group._inner_steps
    )
    assert len(calibrate_group._inner_steps) == 4
    assert all(
        isinstance(s, ChannelSplitter)
        for s in split_group._inner_steps
    )
    assert len(split_group._inner_steps) == 4


def test_pre_calibration_and_extraction_resumability(synthetic_raw_layout):
    """Re-running with overwrite=False does not re-run already-completed steps."""
    config = _make_config(synthetic_raw_layout, overwrite=False)
    _setup_calibration_files(config)

    target = _target_for(synthetic_raw_layout)
    pre_calibration_and_extraction(config, target)

    layout = DataLayout(config.paths)
    stage1 = layout.stage1_output_folder(
        target.star_name, target.obs_night, target.filter_name,
    )
    cal_path = stage1 / "CALIB_SPHER_SCIENCE_001.fits"
    first_bytes = cal_path.read_bytes()

    # Re-run; the calibrated file should be byte-identical.
    pre_calibration_and_extraction(config, target)
    assert cal_path.read_bytes() == first_bytes


# ----------------------------------------------------------------------
# Error paths
# ----------------------------------------------------------------------
def test_pre_calibration_and_extraction_missing_target_raises(synthetic_raw_layout):
    config = _make_config(synthetic_raw_layout)
    with pytest.raises(InvalidConfigError):
        pre_calibration_and_extraction(config, None)


def test_pre_calibration_and_extraction_missing_star_input_raises(
    synthetic_raw_layout,
):
    config = _make_config(synthetic_raw_layout)
    bad = StarTarget(
        star_name="DOES_NOT_EXIST",
        obs_night="2020-01-01",
        filter_name="CI_H",
    )
    with pytest.raises(StepExecutionError):
        pre_calibration_and_extraction(config, bad)


def test_pre_calibration_and_extraction_no_center_files_skips_step(
    synthetic_raw_layout, caplog,
):
    """If the star has no CENTER files we still build calib + split steps."""
    import os
    # Remove the CENTER file from the synthetic layout.
    star = synthetic_raw_layout["star"]
    obs = synthetic_raw_layout["obs_night"]
    filt = synthetic_raw_layout["filter_name"]
    center_path = (
        synthetic_raw_layout["raw"] / star / "IRDIS" / obs / filt
        / "SPHER_CENTER_001.fits"
    )
    os.remove(center_path)

    config = _make_config(synthetic_raw_layout)
    _setup_calibration_files(config)

    target = _target_for(synthetic_raw_layout)
    pipeline = pre_calibration_and_extraction(config, target)

    # Two parallel groups (calibrate + split) with 3 inner steps each;
    # no CenteringPreparer.
    assert len(pipeline.steps) == 2
    types = [type(s).__name__ for s in pipeline.steps]
    assert "CenteringPreparer" not in types
    assert all(isinstance(s, ParallelStepGroup) for s in pipeline.steps)
    assert len(pipeline.steps[0]._inner_steps) == 3
    assert len(pipeline.steps[1]._inner_steps) == 3
