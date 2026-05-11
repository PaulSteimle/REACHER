"""Tests for the Phase 4 inspection batch helpers."""
from __future__ import annotations

from pathlib import Path

import pytest

from REACHER.batch.inspection import (
    cleanup_all_folders,
    grab_all_files,
    return_csv_files,
)
from REACHER.config import (
    CalibrationConfig,
    PathConfig,
    ProcessingConfig,
    ReacherConfig,
)
from REACHER.discovery.layout import DataLayout, list_directory
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


def _seed_centers_csv(config, star, obs, filt, content="frames\n"):
    """Drop a fake centers.csv (and PNG) into ``stage2/`` for inspection tests."""
    layout = DataLayout(config.paths)
    stage2 = layout.stage2_output_folder(star, obs, filt)
    stage2.mkdir(parents=True, exist_ok=True)
    (stage2 / "centers.csv").write_text(content)
    pngs = stage2 / "recentering_pngs"
    pngs.mkdir(exist_ok=True)
    (pngs / "frame1.png").write_text("PNG_DATA")
    return stage2


# ----------------------------------------------------------------------
# list_directory (re-exported)
# ----------------------------------------------------------------------
def test_list_directory_reexport(tmp_path):
    (tmp_path / "a.fits").write_text("x")
    (tmp_path / ".DS_Store").write_text("x")
    assert list_directory(tmp_path) == ["a.fits"]


# ----------------------------------------------------------------------
# grab_all_files
# ----------------------------------------------------------------------
def test_grab_all_files_copies_centers_csv(synthetic_raw_layout):
    config = _make_config(synthetic_raw_layout)
    star = synthetic_raw_layout["star"]
    obs = synthetic_raw_layout["obs_night"]
    filt = synthetic_raw_layout["filter_name"]

    _seed_centers_csv(config, star, obs, filt, "frames\nABC.fits\n")

    created = grab_all_files(config, [star])
    assert len(created) == 1
    dest = created[0]
    assert (dest / "centers.csv").exists()
    assert (dest / "centers.csv").read_text().startswith("frames")
    assert (dest / "frame1.png").exists()


def test_grab_all_files_skips_targets_without_csv(synthetic_raw_layout):
    """Targets without a stage2/centers.csv should not error."""
    config = _make_config(synthetic_raw_layout)
    # We never seed a centers.csv -> no work to do.
    Path(synthetic_raw_layout["reduced"], "HD_TEST", "IRDIS", "2020-01-01", "CI_H").mkdir(
        parents=True, exist_ok=True,
    )
    created = grab_all_files(config, ["HD_TEST"])
    assert created == []


def test_grab_all_files_handles_missing_star(synthetic_raw_layout):
    """A star not present in reduced_output_dir should be ignored."""
    config = _make_config(synthetic_raw_layout)
    created = grab_all_files(config, ["does_not_exist"])
    assert created == []


def test_grab_all_files_requires_config():
    with pytest.raises(InvalidConfigError):
        grab_all_files(None, [])


# ----------------------------------------------------------------------
# return_csv_files
# ----------------------------------------------------------------------
def test_return_csv_files_reverses_grab(synthetic_raw_layout):
    config = _make_config(synthetic_raw_layout)
    star = synthetic_raw_layout["star"]
    obs = synthetic_raw_layout["obs_night"]
    filt = synthetic_raw_layout["filter_name"]
    _seed_centers_csv(config, star, obs, filt, "frames\nA.fits,1,2,3,4,5,6,7,8\n")
    grab_all_files(config, [star])

    # Edit the recentering centers.csv (as the operator would).
    layout = DataLayout(config.paths)
    edited = (
        layout.recentering_folder
        / f"{star}_{obs}_{filt}"
        / "centers.csv"
    )
    edited.write_text("frames\nA.fits,99,99,99,99,99,99,99,99\n")

    written = return_csv_files(config, [star])
    assert len(written) == 1
    stage2_csv = (
        layout.stage2_output_folder(star, obs, filt) / "centers.csv"
    )
    assert "99" in stage2_csv.read_text()


def test_return_csv_files_skips_when_no_recentering_dir(synthetic_raw_layout):
    """No edited centers.csv at the recentering folder => no-op."""
    config = _make_config(synthetic_raw_layout)
    star = synthetic_raw_layout["star"]
    obs = synthetic_raw_layout["obs_night"]
    filt = synthetic_raw_layout["filter_name"]
    # Create the reduced layout but do not seed any recentering dir.
    Path(synthetic_raw_layout["reduced"], star, "IRDIS", obs, filt).mkdir(
        parents=True, exist_ok=True,
    )
    written = return_csv_files(config, [star])
    assert written == []


# ----------------------------------------------------------------------
# cleanup_all_folders
# ----------------------------------------------------------------------
def _seed_calibration_dirs(config):
    layout = DataLayout(config.paths)
    for folder in (
        layout.bad_pix_mask_folder,
        layout.master_dark_folder,
        layout.master_flat_folder,
        layout.master_background_folder,
        layout.recentering_folder,
    ):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "x.fits").write_text("data")


def test_cleanup_all_folders_clean_calibrations(synthetic_raw_layout):
    config = _make_config(synthetic_raw_layout)
    config.processing.clean_calibrations = True
    _seed_calibration_dirs(config)

    cleanup_all_folders(config)
    layout = DataLayout(config.paths)
    assert not layout.bad_pix_mask_folder.exists()
    assert not layout.master_dark_folder.exists()
    assert not layout.master_flat_folder.exists()
    assert not layout.master_background_folder.exists()
    assert not layout.recentering_folder.exists()


def test_cleanup_all_folders_clean_stage_1_and_2(synthetic_raw_layout):
    config = _make_config(synthetic_raw_layout)
    config.processing.clean_stage_1_and_2 = True

    layout = DataLayout(config.paths)
    star = synthetic_raw_layout["star"]
    stage1 = layout.stage1_output_folder(
        star, synthetic_raw_layout["obs_night"],
        synthetic_raw_layout["filter_name"],
    )
    stage2 = layout.stage2_output_folder(
        star, synthetic_raw_layout["obs_night"],
        synthetic_raw_layout["filter_name"],
    )
    stage1.mkdir(parents=True, exist_ok=True)
    stage2.mkdir(parents=True, exist_ok=True)
    (stage1 / "x.fits").write_text("data")
    (stage2 / "x.fits").write_text("data")

    cleanup_all_folders(config)
    assert not stage1.exists()
    assert not stage2.exists()


def test_cleanup_all_folders_does_nothing_when_flags_false(synthetic_raw_layout):
    """Default flags off -> existing files survive."""
    config = _make_config(synthetic_raw_layout)
    _seed_calibration_dirs(config)
    layout = DataLayout(config.paths)
    cleanup_all_folders(config)
    assert layout.master_dark_folder.exists()


def test_cleanup_all_folders_does_not_use_os_system(monkeypatch, synthetic_raw_layout):
    """Architecture Rule 5: shutil.rmtree, never os.system."""
    config = _make_config(synthetic_raw_layout)
    config.processing.clean_calibrations = True
    _seed_calibration_dirs(config)

    called = {"os_system": False}

    def fake_system(cmd):
        called["os_system"] = True
        return 0

    monkeypatch.setattr("os.system", fake_system)
    cleanup_all_folders(config)
    assert not called["os_system"]


def test_cleanup_all_folders_requires_config():
    with pytest.raises(InvalidConfigError):
        cleanup_all_folders(None)
