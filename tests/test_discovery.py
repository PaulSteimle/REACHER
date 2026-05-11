"""Tests for the Phase 4 dataset-discovery layer."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from REACHER.config import (
    CalibrationConfig,
    PathConfig,
    ProcessingConfig,
    ReacherConfig,
)
from REACHER.discovery import (
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
from REACHER.exceptions import (
    CalibrationFileNotFoundError,
    DataFrameError,
)


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


# ----------------------------------------------------------------------
# StarTarget
# ----------------------------------------------------------------------
def test_star_target_is_frozen():
    t = StarTarget(star_name="A", obs_night="2020-01-01", filter_name="H")
    with pytest.raises(Exception):
        t.star_name = "B"  # type: ignore[misc]


def test_star_target_with_overrides():
    t = StarTarget(star_name="A", obs_night="2020-01-01", filter_name="H")
    t2 = t.with_overrides(
        star_center_index=1,
        fixed_user_mastersky_left="/tmp/sky.fits",
        sky_files_exist=True,
    )
    assert t2.star_center_index == 1
    assert t2.fixed_user_mastersky_left == "/tmp/sky.fits"
    assert t2.sky_files_exist is True
    # original is untouched
    assert t.star_center_index == 0


def test_star_target_fixed_user_mastersky_for():
    t = StarTarget(
        star_name="A", obs_night="x", filter_name="H",
        fixed_user_mastersky_left="L.fits",
        fixed_user_mastersky_right="R.fits",
    )
    assert t.fixed_user_mastersky_for("LEFT") == "L.fits"
    assert t.fixed_user_mastersky_for("RIGHT") == "R.fits"
    with pytest.raises(ValueError):
        t.fixed_user_mastersky_for("UP")


# ----------------------------------------------------------------------
# DataLayout
# ----------------------------------------------------------------------
def test_data_layout_paths(synthetic_raw_layout):
    config = _make_config(synthetic_raw_layout)
    layout = DataLayout(config.paths)

    assert layout.bad_pix_mask_folder == Path(config.paths.reduced_output_dir) / "bad_pix_masks"
    assert layout.master_dark_folder == Path(config.paths.reduced_output_dir) / "master_dark"
    assert layout.master_flat_folder == Path(config.paths.reduced_output_dir) / "master_flat"
    assert layout.bad_pix_mask_filepath("2020-01-01").name == "bad_pix_mask_2020-01-01.fits"
    assert layout.master_dark_filepath("2020-01-01", 1.0).name == "master_dark_2020-01-01_1.0.fits"
    assert layout.master_flat_filepath("2020-01-01", "CI_H").name == "master_flat_2020-01-01_CI_H.fits"
    assert layout.clump_mask_filepath.name == "IRDIS_clump_mask.fits"

    star_in = layout.star_input_folder("HD_TEST", "2020-01-01", "CI_H")
    assert star_in.parts[-4:] == ("HD_TEST", "IRDIS", "2020-01-01", "CI_H")

    assert layout.left_output_folder("HD_TEST", "2020-01-01", "CI_H").parts[-2:] == ("left", "raw_files")
    assert layout.right_output_folder("HD_TEST", "2020-01-01", "CI_H").parts[-2:] == ("right", "raw_files")


def test_data_layout_channel_output_folder(synthetic_raw_layout):
    config = _make_config(synthetic_raw_layout)
    layout = DataLayout(config.paths)

    assert (
        layout.channel_output_folder("HD", "2020", "H", "LEFT")
        == layout.left_output_folder("HD", "2020", "H")
    )
    assert (
        layout.channel_output_folder("HD", "2020", "H", "RIGHT")
        == layout.right_output_folder("HD", "2020", "H")
    )
    with pytest.raises(ValueError):
        layout.channel_output_folder("HD", "2020", "H", "UP")


def test_data_layout_master_sky_filepath_overridable(synthetic_raw_layout):
    config = _make_config(synthetic_raw_layout)
    layout = DataLayout(config.paths)

    default = layout.master_sky_filepath("HD", "2020", "H", "LEFT")
    assert default.name == "master_sky_HD_LEFT.fits"

    user = layout.master_sky_filepath(
        "HD", "2020", "H", "LEFT", fixed_user_mastersky="/tmp/x.fits",
    )
    assert str(user) == "/tmp/x.fits"


def test_list_directory_drops_ds_store(tmp_path):
    (tmp_path / "real_file.fits").write_text("x")
    (tmp_path / ".DS_Store").write_text("x")
    (tmp_path / "._.DS_Store").write_text("x")
    entries = list_directory(tmp_path)
    assert sorted(entries) == ["real_file.fits"]


def test_list_directory_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_directory(tmp_path / "missing")


# ----------------------------------------------------------------------
# DataFrame builders
# ----------------------------------------------------------------------
def test_make_dark_DF_finds_two_exptimes(synthetic_raw_layout):
    df = make_dark_DF(synthetic_raw_layout["raw"])
    assert not df.empty
    assert set(df.columns) >= {
        "filename", "obs_night", "obs_run", "filter", "exp_time",
        "filepath", "nd_filter",
    }
    # Both 1.0 and 4.0 exposure times should be discovered.
    assert sorted(df["exp_time"].unique().tolist()) == [1.0, 4.0]
    assert df["obs_night"].unique().tolist() == ["2020-01-01"]


def test_make_flat_DF_includes_synthetic_flat(synthetic_raw_layout):
    df = make_flat_DF(synthetic_raw_layout["raw"])
    assert len(df) == 1
    assert df.iloc[0]["obs_night"] == "2020-01-01"


def test_make_background_DF_empty_when_no_dir(synthetic_raw_layout):
    df = make_background_DF(synthetic_raw_layout["raw"])
    assert df.empty


def test_make_star_DF_finds_files(synthetic_raw_layout):
    df = make_star_DF(
        synthetic_raw_layout["raw"],
        synthetic_raw_layout["star"],
        synthetic_raw_layout["obs_night"],
        synthetic_raw_layout["filter_name"],
    )
    assert len(df) == 4
    assert set(df["type"].unique()) == {"OBJECT", "OBJECT,FLUX", "OBJECT,CENTER", "SKY"}


# ----------------------------------------------------------------------
# Star iteration
# ----------------------------------------------------------------------
def test_list_target_stars_excludes_calib_dirs(synthetic_raw_layout):
    stars = list_target_stars(synthetic_raw_layout["raw"])
    assert stars == ["HD_TEST"]


def test_iter_star_targets_skips_nd_filters(synthetic_raw_layout):
    targets = list(iter_star_targets(
        synthetic_raw_layout["raw"], ["HD_TEST"],
    ))
    assert len(targets) == 1
    assert targets[0].star_name == "HD_TEST"
    assert targets[0].filter_name == "CI_H"


def test_iter_star_targets_includes_nd_when_requested(synthetic_raw_layout):
    targets = list(iter_star_targets(
        synthetic_raw_layout["raw"], ["HD_TEST"],
        skip_nd_filters=False,
    ))
    filter_names = sorted(t.filter_name for t in targets)
    assert filter_names == ["CI_H", "ND_H"]


def test_iter_star_targets_warns_on_missing_star(synthetic_raw_layout, caplog):
    targets = list(iter_star_targets(
        synthetic_raw_layout["raw"], ["DOES_NOT_EXIST"],
    ))
    assert targets == []


# ----------------------------------------------------------------------
# stars_from_ref_cubes / iter_star_targets dict input
# ----------------------------------------------------------------------
def test_stars_from_ref_cubes_str_value():
    ref = {"HD377": "HD342"}
    assert stars_from_ref_cubes(ref) == ["HD342", "HD377"]


def test_stars_from_ref_cubes_none_value():
    ref = {"HD22049": None}
    assert stars_from_ref_cubes(ref) == ["HD22049"]


def test_stars_from_ref_cubes_dict_value_per_obs_night():
    ref = {
        "HD207129": {
            "2022-10-14": "HD207400",
            "2023-10-05": "HD206690",
            "2024-01-01": None,
        },
    }
    assert stars_from_ref_cubes(ref) == ["HD206690", "HD207129", "HD207400"]


def test_stars_from_ref_cubes_mixed_schema():
    ref = {
        "HD377": "HD342",
        "HD22049": None,
        "HD207129": {"2022-10-14": "HD207400", "2023-10-05": "HD206690"},
    }
    assert stars_from_ref_cubes(ref) == [
        "HD206690", "HD207129", "HD207400", "HD22049", "HD342", "HD377",
    ]


def test_stars_from_ref_cubes_dedups_overlap():
    ref = {"HD377": "HD22049", "HD22049": None}
    assert stars_from_ref_cubes(ref) == ["HD22049", "HD377"]


def test_stars_from_ref_cubes_rejects_invalid_value():
    from REACHER.exceptions import InvalidConfigError

    with pytest.raises(InvalidConfigError):
        stars_from_ref_cubes({"HD377": 123})


def test_iter_star_targets_accepts_ref_cubes_dict(synthetic_raw_layout):
    targets = list(iter_star_targets(
        synthetic_raw_layout["raw"], {"HD_TEST": None},
    ))
    assert len(targets) == 1
    assert targets[0].star_name == "HD_TEST"


# ----------------------------------------------------------------------
# Calibration resolver
# ----------------------------------------------------------------------
def test_resolve_star_calibration_exact_match(synthetic_raw_layout):
    config = _make_config(synthetic_raw_layout)
    layout = DataLayout(config.paths)

    # Synthesize calibration product files on disk.
    layout.bad_pix_mask_folder.mkdir(parents=True, exist_ok=True)
    layout.master_dark_folder.mkdir(parents=True, exist_ok=True)
    layout.master_flat_folder.mkdir(parents=True, exist_ok=True)

    badpix = layout.bad_pix_mask_filepath("2020-01-01")
    badpix.write_bytes(b"")
    masterdark = layout.master_dark_filepath("2020-01-01", 1.0)
    masterdark.write_bytes(b"")
    masterflat = layout.master_flat_filepath("2020-01-01", "CI_H")
    masterflat.write_bytes(b"")

    star_df = make_star_DF(
        config.paths.raw_input_dir, "HD_TEST", "2020-01-01", "CI_H",
    )
    out = resolve_star_calibration(star_df, config, calib_type="science")

    assert (out["bad_pix_mask"] == str(badpix)).all()
    assert (out["master_dark"] == str(masterdark)).all()
    assert (out["master_flat"] == str(masterflat)).all()


def test_resolve_star_calibration_nearest_within_threshold(synthetic_raw_layout):
    """If the exact-night calibration is missing, fall back to nearest."""
    config = _make_config(synthetic_raw_layout)
    layout = DataLayout(config.paths)
    layout.bad_pix_mask_folder.mkdir(parents=True, exist_ok=True)
    layout.master_dark_folder.mkdir(parents=True, exist_ok=True)
    layout.master_flat_folder.mkdir(parents=True, exist_ok=True)

    # Provide a 2020-01-03 calibration set (3 days off, well within 8).
    near = "2020-01-03"
    (layout.bad_pix_mask_folder / f"bad_pix_mask_{near}.fits").write_bytes(b"")
    (layout.master_dark_folder / f"master_dark_{near}_1.0.fits").write_bytes(b"")
    (layout.master_flat_folder / f"master_flat_{near}_CI_H.fits").write_bytes(b"")

    star_df = make_star_DF(
        config.paths.raw_input_dir, "HD_TEST", "2020-01-01", "CI_H",
    )
    out = resolve_star_calibration(star_df, config, calib_type="science")
    assert (out["bad_pix_mask"].str.contains("2020-01-03")).all()
    assert (out["master_dark"].str.contains("2020-01-03")).all()
    assert (out["master_flat"].str.contains("2020-01-03")).all()


def test_resolve_star_calibration_beyond_threshold_raises(synthetic_raw_layout):
    """No calibration within 8 days -> CalibrationFileNotFoundError."""
    config = _make_config(synthetic_raw_layout)
    layout = DataLayout(config.paths)
    layout.bad_pix_mask_folder.mkdir(parents=True, exist_ok=True)

    # Put a calibration 100 days away.
    (layout.bad_pix_mask_folder / "bad_pix_mask_2020-04-15.fits").write_bytes(b"")

    star_df = make_star_DF(
        config.paths.raw_input_dir, "HD_TEST", "2020-01-01", "CI_H",
    )
    with pytest.raises(CalibrationFileNotFoundError):
        resolve_star_calibration(star_df, config, calib_type="science")


def test_resolve_star_calibration_master_flat_stage_keeps_missing_dark(synthetic_raw_layout):
    """Stage 'master_flat' tolerates missing master darks (legacy behaviour)."""
    config = _make_config(synthetic_raw_layout)
    layout = DataLayout(config.paths)
    layout.bad_pix_mask_folder.mkdir(parents=True, exist_ok=True)

    # Bad pix mask exists; master dark does not.
    (layout.bad_pix_mask_folder / "bad_pix_mask_2020-01-01.fits").write_bytes(b"")

    flat_df = make_flat_DF(config.paths.raw_input_dir)
    out = resolve_star_calibration(flat_df, config, calib_type="master_flat")
    assert "master_dark" in out.columns
    assert (out["master_dark"] == "None").all()


def test_resolve_star_calibration_unknown_calib_type(synthetic_raw_layout):
    config = _make_config(synthetic_raw_layout)
    star_df = make_star_DF(
        config.paths.raw_input_dir, "HD_TEST", "2020-01-01", "CI_H",
    )
    with pytest.raises(KeyError):
        resolve_star_calibration(star_df, config, calib_type="not_a_stage")


def test_resolve_star_calibration_missing_columns_raises(synthetic_raw_layout):
    config = _make_config(synthetic_raw_layout)
    bad = pd.DataFrame({"foo": [1]})
    with pytest.raises(DataFrameError):
        resolve_star_calibration(bad, config, calib_type="science")
