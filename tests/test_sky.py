"""Tests for MasterSkyCreator step."""
import numpy as np
import pytest
from astropy.io import fits

from REACHER.config import PathConfig, ProcessingConfig, ReacherConfig
from REACHER.exceptions import (
    HeaderKeyError,
    InvalidConfigError,
    StepExecutionError,
)
from REACHER.steps.sky import MasterSkyCreator


def _make_config(tmp_path, overwrite):
    raw_dir = tmp_path / "raw"
    reduced_dir = tmp_path / "reduced"
    raw_dir.mkdir(parents=True, exist_ok=True)
    reduced_dir.mkdir(parents=True, exist_ok=True)
    paths = PathConfig(
        raw_input_dir=str(raw_dir), reduced_output_dir=str(reduced_dir)
    )
    processing = ProcessingConfig(overwrite=overwrite)
    return ReacherConfig(paths=paths, processing=processing)


def _hdr_with_temp(base_header, temp):
    """Return a header with the temperature keyword overridden."""
    h = base_header.copy()
    h["HIERARCH ESO INS4 TEMP449 VAL"] = float(temp)
    return h


def test_master_sky_2d_inputs(tmp_path, minimal_sphere_header):
    """Two 2D sky frames -> median per pixel; provenance keywords written."""
    config = _make_config(tmp_path, overwrite=True)

    sky_a = np.full((16, 16), 4.0, dtype=np.float32)
    sky_b = np.full((16, 16), 6.0, dtype=np.float32)

    p_a = tmp_path / "sky_a.fits"
    p_b = tmp_path / "sky_b.fits"
    fits.writeto(p_a, sky_a, header=_hdr_with_temp(minimal_sphere_header, 290.0), overwrite=True)
    fits.writeto(p_b, sky_b, header=_hdr_with_temp(minimal_sphere_header, 292.0), overwrite=True)

    out = tmp_path / "master_sky.fits"
    MasterSkyCreator(
        config=config,
        sky_files=[p_a, p_b],
        master_sky_path=out,
    ).run()

    result = fits.getdata(out)
    expected = np.median(np.stack([sky_a, sky_b]), axis=0)
    assert result.shape == (16, 16)
    assert np.allclose(result, expected)

    hdr = fits.getheader(out)
    assert hdr["MEANTEMP"] == pytest.approx(291.0)
    # Per-file temperatures stored under SKYTMP0..N (legacy SKYTEMP was
    # overwritten in a loop; we keep all of them).
    assert hdr["SKYTMP0"] == pytest.approx(290.0)
    assert hdr["SKYTMP1"] == pytest.approx(292.0)
    assert hdr["SKY0"] == p_a.name
    assert hdr["SKY1"] == p_b.name
    assert "MASTER SKY" in [str(h).strip() for h in hdr.get("HISTORY", [])]


def test_master_sky_3d_collapse(tmp_path, minimal_sphere_header):
    """A 3D sky cube is collapsed to its nanmedian along axis 0 first."""
    config = _make_config(tmp_path, overwrite=True)

    cube = np.stack(
        [
            np.full((8, 8), 5.0, dtype=np.float32),
            np.full((8, 8), 7.0, dtype=np.float32),
            np.full((8, 8), 9.0, dtype=np.float32),
        ]
    )
    expected = np.nanmedian(cube, axis=0)  # 7.0 everywhere

    sky_path = tmp_path / "sky_cube.fits"
    fits.writeto(
        sky_path, cube,
        header=_hdr_with_temp(minimal_sphere_header, 285.0),
        overwrite=True,
    )

    out = tmp_path / "master_sky_3d.fits"
    MasterSkyCreator(
        config=config,
        sky_files=[sky_path],
        master_sky_path=out,
    ).run()

    result = fits.getdata(out)
    assert result.shape == (8, 8)
    assert np.allclose(result, expected)

    hdr = fits.getheader(out)
    assert hdr["MEANTEMP"] == pytest.approx(285.0)


def test_master_sky_missing_input(tmp_path):
    """Missing sky file -> StepExecutionError."""
    config = _make_config(tmp_path, overwrite=True)
    out = tmp_path / "master_sky.fits"
    step = MasterSkyCreator(
        config=config,
        sky_files=[tmp_path / "ghost.fits"],
        master_sky_path=out,
    )
    with pytest.raises(StepExecutionError):
        step.run()
    assert not out.exists()


def test_master_sky_missing_temp_keyword(tmp_path):
    """Sky file without temperature keyword -> HeaderKeyError."""
    config = _make_config(tmp_path, overwrite=True)

    bare_header = fits.Header()
    bare_header["EXPTIME"] = 1.0
    sky_path = tmp_path / "bad_sky.fits"
    fits.writeto(sky_path, np.zeros((4, 4), dtype=np.float32),
                 header=bare_header, overwrite=True)

    out = tmp_path / "master_sky.fits"
    step = MasterSkyCreator(
        config=config,
        sky_files=[sky_path],
        master_sky_path=out,
    )
    with pytest.raises(HeaderKeyError):
        step.run()


def test_master_sky_resumability(tmp_path, minimal_sphere_header):
    """Pre-existing master sky + overwrite=False -> file untouched."""
    config = _make_config(tmp_path, overwrite=False)

    sky = np.full((8, 8), 1.0, dtype=np.float32)
    sky_path = tmp_path / "sky.fits"
    fits.writeto(
        sky_path, sky,
        header=_hdr_with_temp(minimal_sphere_header, 290.0),
        overwrite=True,
    )

    out = tmp_path / "master_sky_existing.fits"
    sentinel = np.full((8, 8), 99.0, dtype=np.float32)
    fits.writeto(out, sentinel, overwrite=True)

    MasterSkyCreator(
        config=config,
        sky_files=[sky_path],
        master_sky_path=out,
    ).run()

    assert np.allclose(fits.getdata(out), sentinel)


def test_master_sky_empty_input(tmp_path):
    """Empty sky list -> InvalidConfigError on instantiation+run."""
    config = _make_config(tmp_path, overwrite=True)
    out = tmp_path / "master_sky.fits"
    with pytest.raises(InvalidConfigError):
        MasterSkyCreator(
            config=config,
            sky_files=[],
            master_sky_path=out,
        ).run()
