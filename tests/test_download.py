"""Tests for the pre-pipeline FITS organizer (utils/download.py)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from REACHER.utils.download import move_files_from_download_dir


def _write_spher_fits(
    path: Path,
    *,
    obj: str,
    date_obs: str,
    filter_name: str = "DB_H23",
    nd_filter: str = "OPEN",
    dpr_type: str = "OBJECT",
    exp_time: float = 64.0,
) -> None:
    hdu = fits.PrimaryHDU(np.zeros((4, 4), dtype=np.float32))
    hdu.header["OBJECT"] = obj
    hdu.header["DATE-OBS"] = date_obs
    hdu.header["EXPTIME"] = exp_time
    hdu.header["HIERARCH ESO INS COMB IFLT"] = filter_name
    hdu.header["HIERARCH ESO INS4 FILT2 NAME"] = nd_filter
    hdu.header["HIERARCH ESO DPR TYPE"] = dpr_type
    hdu.writeto(path, overwrite=True)


def test_move_files_organizes_layout(tmp_path):
    download = tmp_path / "download"
    new_root = tmp_path / "raw"
    download.mkdir()

    _write_spher_fits(
        download / "SPHER.2022-10-14T03:12:11.fits",
        obj="HD377",
        date_obs="2022-10-14T03:12:11",
        dpr_type="OBJECT",
    )
    _write_spher_fits(
        download / "SPHER.2022-10-14T22:00:00.fits",
        obj="HD377",
        date_obs="2022-10-14T22:00:00",
        dpr_type="DARK",
    )

    move_files_from_download_dir(str(download), str(new_root), {})

    pre_midnight_morning = (
        new_root / "HD377" / "IRDIS" / "2022-10-13" / "DB_H23"
        / "SCIENCE_SPHER.2022-10-14T03:12:11.fits"
    )
    after_dusk = (
        new_root / "HD377" / "IRDIS" / "2022-10-14" / "DB_H23"
        / "DARK_SPHER.2022-10-14T22:00:00.fits"
    )
    assert pre_midnight_morning.exists()
    assert after_dusk.exists()


def test_move_files_applies_rename_dict(tmp_path):
    download = tmp_path / "download"
    new_root = tmp_path / "raw"
    download.mkdir()

    _write_spher_fits(
        download / "SPHER.2023-05-01T22:00:00.fits",
        obj="epsEri",
        date_obs="2023-05-01T22:00:00",
        dpr_type="OBJECT",
    )

    move_files_from_download_dir(
        str(download), str(new_root),
        {"epsEri": "HD22049_epsEri"},
    )

    expected = (
        new_root / "HD22049_epsEri" / "IRDIS" / "2023-05-01" / "DB_H23"
        / "SCIENCE_SPHER.2023-05-01T22:00:00.fits"
    )
    assert expected.exists()


def test_move_files_skips_generic_object_header(tmp_path):
    download = tmp_path / "download"
    new_root = tmp_path / "raw"
    download.mkdir()

    _write_spher_fits(
        download / "SPHER.weird.fits",
        obj="OBJECT",
        date_obs="2023-01-01T22:00:00",
        dpr_type="OBJECT",
    )

    move_files_from_download_dir(str(download), str(new_root), {})

    assert not any(new_root.rglob("*.fits"))


def test_move_files_no_input_files_warns(tmp_path, caplog):
    download = tmp_path / "download"
    new_root = tmp_path / "raw"
    download.mkdir()

    move_files_from_download_dir(str(download), str(new_root), {})
    assert any("no FITS files" in rec.message for rec in caplog.records)


def test_move_files_strips_dark_lamp_tokens(tmp_path):
    download = tmp_path / "download"
    new_root = tmp_path / "raw"
    download.mkdir()

    _write_spher_fits(
        download / "SPHER.lamp.fits",
        obj="HD123",
        date_obs="2023-01-01T22:00:00",
        dpr_type="DARK,LAMP",
    )

    move_files_from_download_dir(str(download), str(new_root), {})

    expected = (
        new_root / "HD123" / "IRDIS" / "2023-01-01" / "DB_H23"
        / "LAMP_SPHER.lamp.fits"
    )
    assert expected.exists()
