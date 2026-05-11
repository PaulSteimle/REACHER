"""Tests for the Phase 4 centering_and_bad_pix_correction batch helper.

The cube-assembly + flux + sky steps depend on heavy numerical kernels
(VIP-HCI, photutils, scipy). The batch helper itself is the
build-and-wire glue we exercise here; the per-step numerical tests
live in tests/test_cube_assembly.py / test_flux.py / test_sky.py.

These tests therefore focus on the structural contract: pipeline
wiring (number + types of steps), ref-star handling, and short-circuit
behaviour (missing CENTER files, only_flux flag, ...).
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from REACHER.batch.calibration import (
    make_all_masterdarks,
    make_all_masterflats,
    make_bad_pix_masks,
)
from REACHER.batch.cubes import (
    _resolve_ref_stars,
    centering_and_bad_pix_correction,
)
from REACHER.batch.extraction import pre_calibration_and_extraction
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
        paths=paths, processing=processing,
        calibration=CalibrationConfig(),
    )


def _target_for(layout_dict) -> StarTarget:
    return StarTarget(
        star_name=layout_dict["star"],
        obs_night=layout_dict["obs_night"],
        filter_name=layout_dict["filter_name"],
    )


def _setup_pre_calibration(synthetic_raw_layout):
    """Run the full calibration + extraction chain to populate stage1/stage2."""
    config = _make_config(synthetic_raw_layout)
    make_bad_pix_masks(config)
    make_all_masterdarks(config)
    make_all_masterflats(config)
    target = _target_for(synthetic_raw_layout)
    pre_calibration_and_extraction(config, target)
    return config, target


# ----------------------------------------------------------------------
# _resolve_ref_stars unit
# ----------------------------------------------------------------------
def test_resolve_ref_stars_none_when_no_dict():
    target = StarTarget(star_name="A", obs_night="x", filter_name="H")
    assert _resolve_ref_stars(None, target) == []
    assert _resolve_ref_stars({}, target) == []
    assert _resolve_ref_stars({"OTHER": "REF"}, target) == []


def test_resolve_ref_stars_string():
    target = StarTarget(star_name="A", obs_night="x", filter_name="H")
    assert _resolve_ref_stars({"A": "REF"}, target) == ["REF"]


def test_resolve_ref_stars_list():
    target = StarTarget(star_name="A", obs_night="x", filter_name="H")
    assert _resolve_ref_stars({"A": ["R1", "R2"]}, target) == ["R1", "R2"]


def test_resolve_ref_stars_dict_keyed_on_obs_night():
    target = StarTarget(
        star_name="A", obs_night="2022-10-14", filter_name="H",
    )
    mapping = {"A": {"2022-10-14": "REF1", "2023-10-05": "REF2"}}
    assert _resolve_ref_stars(mapping, target) == ["REF1"]


def test_resolve_ref_stars_dict_missing_obs_night():
    target = StarTarget(star_name="A", obs_night="x", filter_name="H")
    mapping = {"A": {"y": "REF"}}
    assert _resolve_ref_stars(mapping, target) == []


# ----------------------------------------------------------------------
# Top-level helper -- wiring contract
# ----------------------------------------------------------------------
def test_centering_and_bad_pix_correction_requires_target(synthetic_raw_layout):
    config = _make_config(synthetic_raw_layout)
    with pytest.raises(InvalidConfigError):
        centering_and_bad_pix_correction(config, None)


def test_centering_and_bad_pix_correction_missing_stage2_raises(synthetic_raw_layout):
    config = _make_config(synthetic_raw_layout)
    target = _target_for(synthetic_raw_layout)
    # We did not run pre_calibration_and_extraction => stage2 is missing.
    with pytest.raises(StepExecutionError):
        centering_and_bad_pix_correction(config, target)


def test_centering_and_bad_pix_correction_returns_empty_when_no_center_files(
    synthetic_raw_layout, monkeypatch,
):
    """Targets with no CENTER files yield an empty pipeline (legacy: ref star)."""
    config = _make_config(synthetic_raw_layout)
    target = _target_for(synthetic_raw_layout)

    layout = DataLayout(config.paths)
    stage2 = layout.stage2_output_folder(
        target.star_name, target.obs_night, target.filter_name,
    )
    stage2.mkdir(parents=True, exist_ok=True)
    # no LEFT_*CENTER files: helper short-circuits with an empty pipeline.

    pipeline = centering_and_bad_pix_correction(config, target)
    assert len(pipeline.steps) == 0


def _fake_get_center(*args, **kwargs):
    """Stub for utils.centering.get_center to avoid the VIP dependency."""
    return (0.0, 0.0)


def test_centering_and_bad_pix_correction_wires_sky_flux_cubes(
    synthetic_raw_layout, monkeypatch,
):
    """Run the helper with the heavy kernels stubbed and confirm step counts."""
    config, target = _setup_pre_calibration(synthetic_raw_layout)

    # Stub get_center -- imported by REACHER.batch.cubes at module load time.
    monkeypatch.setattr(
        "REACHER.batch.cubes.get_center", _fake_get_center,
    )

    # Patch CubeAssembler.run + FluxCalibrator.run + MasterSkyCreator.run
    # to no-ops so we only test wiring.
    monkeypatch.setattr(
        "REACHER.steps.cube_assembly.CubeAssembler.run", lambda self: None,
    )
    monkeypatch.setattr(
        "REACHER.steps.flux.FluxCalibrator.run", lambda self: None,
    )
    monkeypatch.setattr(
        "REACHER.steps.sky.MasterSkyCreator.run", lambda self: None,
    )

    pipeline = centering_and_bad_pix_correction(config, target)
    types = [type(s).__name__ for s in pipeline.steps]

    # 2 master-sky steps (LEFT + RIGHT), 2 flux steps (one per channel),
    # 2 cube assembly steps (one per channel). Total = 6.
    assert types.count("MasterSkyCreator") == 2
    assert types.count("FluxCalibrator") == 2
    assert types.count("CubeAssembler") == 2


def test_centering_and_bad_pix_correction_only_flux_skips_cubes(
    synthetic_raw_layout, monkeypatch,
):
    """only_flux=True: pipeline contains no CubeAssembler steps."""
    config, target = _setup_pre_calibration(synthetic_raw_layout)
    config.processing.only_flux = True

    monkeypatch.setattr(
        "REACHER.batch.cubes.get_center", _fake_get_center,
    )
    monkeypatch.setattr(
        "REACHER.steps.flux.FluxCalibrator.run", lambda self: None,
    )
    monkeypatch.setattr(
        "REACHER.steps.sky.MasterSkyCreator.run", lambda self: None,
    )

    pipeline = centering_and_bad_pix_correction(config, target)
    types = [type(s).__name__ for s in pipeline.steps]
    assert "CubeAssembler" not in types
    assert types.count("FluxCalibrator") == 2


def test_centering_and_bad_pix_correction_user_supplied_mastersky(
    synthetic_raw_layout, monkeypatch,
):
    """When ``target.fixed_user_mastersky_*`` is set we skip MasterSkyCreator."""
    config, base_target = _setup_pre_calibration(synthetic_raw_layout)
    fake_sky = synthetic_raw_layout["reduced"] / "user_sky_left.fits"
    fake_sky.write_bytes(b"")
    target = base_target.with_overrides(
        fixed_user_mastersky_left=str(fake_sky),
        fixed_user_mastersky_right=str(fake_sky),
    )

    monkeypatch.setattr(
        "REACHER.batch.cubes.get_center", _fake_get_center,
    )
    monkeypatch.setattr(
        "REACHER.steps.cube_assembly.CubeAssembler.run", lambda self: None,
    )
    monkeypatch.setattr(
        "REACHER.steps.flux.FluxCalibrator.run", lambda self: None,
    )

    pipeline = centering_and_bad_pix_correction(config, target)
    types = [type(s).__name__ for s in pipeline.steps]
    assert "MasterSkyCreator" not in types


def test_centering_and_bad_pix_correction_ref_cubes_dict(
    synthetic_raw_layout, monkeypatch,
):
    """Ref star with stage2 produces an extra pair of CubeAssembler steps."""
    config, target = _setup_pre_calibration(synthetic_raw_layout)

    # Build a synthetic ref star stage2 directory by running
    # pre_calibration_and_extraction for a second star. To keep the test
    # cheap we just symlink the existing star's stage2 under a new name.
    ref_star = "HD_REF"
    layout = DataLayout(config.paths)
    real_stage2 = layout.stage2_output_folder(
        target.star_name, target.obs_night, target.filter_name,
    )
    ref_stage2 = layout.stage2_output_folder(
        ref_star, target.obs_night, target.filter_name,
    )
    ref_stage2.parent.mkdir(parents=True, exist_ok=True)
    # copy the directory tree.
    import shutil
    shutil.copytree(real_stage2, ref_stage2)

    monkeypatch.setattr(
        "REACHER.batch.cubes.get_center", _fake_get_center,
    )
    monkeypatch.setattr(
        "REACHER.steps.cube_assembly.CubeAssembler.run", lambda self: None,
    )
    monkeypatch.setattr(
        "REACHER.steps.flux.FluxCalibrator.run", lambda self: None,
    )
    monkeypatch.setattr(
        "REACHER.steps.sky.MasterSkyCreator.run", lambda self: None,
    )

    ref_cubes = {target.star_name: ref_star}
    pipeline = centering_and_bad_pix_correction(
        config, target, ref_cubes=ref_cubes,
    )
    types = [type(s).__name__ for s in pipeline.steps]
    # 2 science cubes (LEFT + RIGHT) + 2 ref cubes (LEFT + RIGHT) = 4.
    assert types.count("CubeAssembler") == 4
