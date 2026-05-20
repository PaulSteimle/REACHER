import numpy as np
import pytest
from astropy.io import fits


@pytest.fixture
def minimal_sphere_header():
    header = fits.Header()
    header["EXPTIME"] = 1.0
    header["DATE-OBS"] = "2020-01-01T00:00:00.000"
    header["DATE"] = "2020-01-01T00:00:00.000"
    header["OBJECT"] = "TEST"
    header["HIERARCH ESO INS COMB IFLT"] = "H"
    header["HIERARCH ESO INS4 FILT2 NAME"] = "ND"
    header["HIERARCH ESO INS4 FILT2 ID"] = "ND"
    header["HIERARCH ESO DPR TYPE"] = "SCIENCE"
    header["HIERARCH ESO TEL PARANG END"] = 0.0
    header["HIERARCH ESO TEL PARANG START"] = 0.0
    header["HIERARCH ESO INS1 DITH POSX"] = 0.0
    header["HIERARCH ESO INS1 DITH POSY"] = 0.0
    header["HIERARCH ESO INS4 TEMP449 VAL"] = 290.0
    return header


@pytest.fixture
def synthetic_full_frame():
    return np.zeros((1024, 2048), dtype=np.float32)


@pytest.fixture
def synthetic_left_frame():
    return np.zeros((995, 875), dtype=np.float32)


@pytest.fixture
def synthetic_right_frame():
    return np.zeros((995, 875), dtype=np.float32)


@pytest.fixture
def synthetic_dark_cube():
    cube = np.zeros((10, 1024, 2048), dtype=np.float32)
    hot_pixels = [(100, 100), (200, 150), (400, 1600)]
    cold_pixels = [(300, 400), (500, 1200)]
    for y, x in hot_pixels:
        cube[:, y, x] = 1000.0
    for y, x in cold_pixels:
        cube[:, y, x] = -1000.0
    return cube, hot_pixels, cold_pixels


@pytest.fixture
def synthetic_flat():
    x = np.linspace(0.9, 1.1, 2048, dtype=np.float32)
    y = np.linspace(0.9, 1.1, 1024, dtype=np.float32)
    gradient = np.outer(y, x)
    return gradient


@pytest.fixture
def synthetic_science_cube():
    cube = np.zeros((5, 1024, 2048), dtype=np.float32)
    cube[:, 512, 1024] = 100.0
    return cube


# ----------------------------------------------------------------------
# Phase 4 fixtures: synthetic on-disk layout
# ----------------------------------------------------------------------
def _write_fits(path, data, header):
    """Write a small FITS file with a SPHERE header into ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(path, np.asarray(data, dtype=np.float32),
                 header=header, overwrite=True)


@pytest.fixture
def synthetic_raw_layout(tmp_path, minimal_sphere_header):
    """Build a minimal raw-input directory tree the discovery layer reads.

    Layout::

        raw/
            DARK/IRDIS/2020-01-01/CI_H/dark1_t1.fits  (1024x2048)
                                       dark1_t4.fits  (1024x2048, exp_time=4)
            FLAT/IRDIS/2020-01-01/CI_H/flat1.fits     (1024x2048)
            HD_TEST/IRDIS/2020-01-01/CI_H/
                SPHER_SCIENCE_001.fits  (3x1024x2048, OBJECT)
                SPHER_FLUX_001.fits     (3x1024x2048, OBJECT,FLUX)
                SPHER_CENTER_001.fits   (3x1024x2048, OBJECT,CENTER)
                SPHER_SKY_001.fits      (3x1024x2048, SKY)
            HD_TEST/IRDIS/2020-01-01/ND_H/
                SPHER_SCIENCE_001.fits

    Calibration files are full IRDIS-shaped (1024x2048) so that the
    BadPixelMaskCreator / MasterDarkCreator / MasterFlatCreator /
    FileCalibrator / ChannelSplitter steps all see acceptable shapes.
    The arrays are filled with simple constants; the test is structural
    (does the pipeline wire the right files together?), not numerical.

    Returns
    -------
    dict
        Layout descriptor keyed on ``raw``, ``reduced``, ``star``,
        ``obs_night``, ``filter_name``, ``header``.
    """
    raw = tmp_path / "raw"
    reduced = tmp_path / "reduced"
    raw.mkdir()
    reduced.mkdir()

    star = "HD_TEST"
    obs_night = "2020-01-01"
    filter_name = "CI_H"

    base_header = minimal_sphere_header.copy()

    irdis_shape = (1024, 2048)
    irdis_cube_shape = (3, 1024, 2048)

    # ---- DARKS (two exp_times under the same obs night) ----
    dark1_header = base_header.copy()
    dark1_header["EXPTIME"] = 1.0
    dark1_header["HIERARCH ESO DPR TYPE"] = "DARK"
    _write_fits(
        raw / "DARK" / "IRDIS" / obs_night / filter_name / "dark1_t1.fits",
        np.full(irdis_shape, 0.5, dtype=np.float32),
        dark1_header,
    )
    dark2_header = dark1_header.copy()
    dark2_header["EXPTIME"] = 4.0
    _write_fits(
        raw / "DARK" / "IRDIS" / obs_night / filter_name / "dark1_t4.fits",
        np.full(irdis_shape, 0.5, dtype=np.float32),
        dark2_header,
    )

    # ---- FLATS ----
    flat_header = base_header.copy()
    flat_header["EXPTIME"] = 1.0
    flat_header["HIERARCH ESO DPR TYPE"] = "FLAT,LAMP"
    _write_fits(
        raw / "FLAT" / "IRDIS" / obs_night / filter_name / "flat1.fits",
        np.full(irdis_shape, 2.0, dtype=np.float32),
        flat_header,
    )

    # ---- STAR FILES ----
    science_header = base_header.copy()
    science_header["EXPTIME"] = 1.0
    science_header["HIERARCH ESO DPR TYPE"] = "OBJECT"
    _write_fits(
        raw / star / "IRDIS" / obs_night / filter_name / "SPHER_SCIENCE_001.fits",
        np.full(irdis_cube_shape, 9.0, dtype=np.float32),
        science_header,
    )
    flux_header = base_header.copy()
    flux_header["EXPTIME"] = 1.0
    flux_header["HIERARCH ESO DPR TYPE"] = "OBJECT,FLUX"
    _write_fits(
        raw / star / "IRDIS" / obs_night / filter_name / "SPHER_FLUX_001.fits",
        np.full(irdis_cube_shape, 9.0, dtype=np.float32),
        flux_header,
    )
    center_header = base_header.copy()
    center_header["EXPTIME"] = 1.0
    center_header["HIERARCH ESO DPR TYPE"] = "OBJECT,CENTER"
    _write_fits(
        raw / star / "IRDIS" / obs_night / filter_name / "SPHER_CENTER_001.fits",
        np.full(irdis_cube_shape, 9.0, dtype=np.float32),
        center_header,
    )
    sky_header = base_header.copy()
    sky_header["EXPTIME"] = 1.0
    sky_header["HIERARCH ESO DPR TYPE"] = "SKY"
    _write_fits(
        raw / star / "IRDIS" / obs_night / filter_name / "SPHER_SKY_001.fits",
        np.full(irdis_cube_shape, 9.0, dtype=np.float32),
        sky_header,
    )

    # ---- STAR FILES under an ND_* filter (should be skipped by iter) ----
    nd_header = science_header.copy()
    _write_fits(
        raw / star / "IRDIS" / obs_night / "ND_H" / "SPHER_SCIENCE_001.fits",
        np.full(irdis_cube_shape, 9.0, dtype=np.float32),
        nd_header,
    )

    return {
        "raw": raw,
        "reduced": reduced,
        "star": star,
        "obs_night": obs_night,
        "filter_name": filter_name,
        "header": base_header,
    }
