"""Tests for CubeAssembler step."""
import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from REACHER.config import (
    CalibrationConfig,
    InstrumentConfig,
    PathConfig,
    ProcessingConfig,
    ReacherConfig,
)
from REACHER.exceptions import (
    HeaderKeyError,
    InvalidConfigError,
    StepExecutionError,
)
from REACHER.steps.cube_assembly import CubeAssembler


def _make_config(
    tmp_path,
    overwrite=True,
    make_plots=False,
    do_cubes=True,
    skip_cosmics=True,
    correct_anamorphism=False,
    skip_dither_correction=True,
    subtract_median_science_sky_bkg=False,
    nproc=1,
):
    raw_dir = tmp_path / "raw"
    reduced_dir = tmp_path / "reduced"
    raw_dir.mkdir(parents=True, exist_ok=True)
    reduced_dir.mkdir(parents=True, exist_ok=True)
    paths = PathConfig(
        raw_input_dir=str(raw_dir), reduced_output_dir=str(reduced_dir)
    )
    processing = ProcessingConfig(
        overwrite=overwrite,
        make_plots=make_plots,
        do_cubes=do_cubes,
        skip_cosmics=skip_cosmics,
        correct_anamorphism=correct_anamorphism,
        skip_dither_correction=skip_dither_correction,
        subtract_median_science_sky_bkg=subtract_median_science_sky_bkg,
        nproc=nproc,
    )
    calibration = CalibrationConfig(
        # tiny test-friendly libraries (no opencv available).
        image_shift_library="ndimage-interp",
        image_shift_interpolation="bicubic",
    )
    instrument = InstrumentConfig()
    return ReacherConfig(
        paths=paths,
        processing=processing,
        calibration=calibration,
        instrument=instrument,
    )


def _make_science_file(
    tmp_path, name, n_frames, base_header,
    angstart=0.0, angend=10.0,
    dith_x=0.0, dith_y=0.0,
    temp=290.0, exptime=1.0, frame_size=128,
    fill_value=1.0,
):
    cube = np.full(
        (n_frames, frame_size, frame_size), fill_value, dtype=np.float32
    )
    h = base_header.copy()
    h["EXPTIME"] = float(exptime)
    h["DATE-OBS"] = "2020-01-01T00:00:00.000"
    h["DATE"] = "2020-01-01T00:01:00.000"
    h["HIERARCH ESO TEL PARANG START"] = float(angstart)
    h["HIERARCH ESO TEL PARANG END"] = float(angend)
    h["HIERARCH ESO INS1 DITH POSX"] = float(dith_x)
    h["HIERARCH ESO INS1 DITH POSY"] = float(dith_y)
    h["HIERARCH ESO INS4 TEMP449 VAL"] = float(temp)
    fits_path = tmp_path / name
    fits.writeto(fits_path, cube, header=h, overwrite=True)
    return fits_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_cube_assembler_happy_path(tmp_path, minimal_sphere_header):
    """End-to-end with all optional steps disabled: outputs are produced."""
    config = _make_config(tmp_path)

    # Two science files of two frames each => 4 frames total.
    f1 = _make_science_file(
        tmp_path, "sci_1.fits", 2, minimal_sphere_header,
        angstart=10.0, angend=12.0, frame_size=63, fill_value=5.0,
    )
    f2 = _make_science_file(
        tmp_path, "sci_2.fits", 2, minimal_sphere_header,
        angstart=12.0, angend=14.0, frame_size=63, fill_value=5.0,
    )

    out_cube = tmp_path / "master_cube.fits"
    out_angs = tmp_path / "parallactic_angles.txt"
    star_center = tmp_path / "centers.csv"

    CubeAssembler(
        config=config,
        science_files=[f1, f2],
        output_file_cube=out_cube,
        output_file_angs=out_angs,
        shift=(0.5, 0.5),
        star_center_file=star_center,
        star_name="TARGET",
        obs_night="2020-01-01",
        filter_name="H",
    ).run()

    assert out_cube.exists()
    assert out_angs.exists()

    angs = np.loadtxt(out_angs)
    assert len(angs) == 4
    # Angles should be in [0, 360) after the SPHERE offset application.
    assert np.all((angs >= 0) & (angs < 360))

    cube = fits.getdata(out_cube)
    # Output cube is padded to 1025x1025 with the integer center shift.
    assert cube.shape == (4, 1025, 1025)

    # CSVs were created alongside.
    assert (tmp_path / "header_infos.csv").exists()
    assert (tmp_path / "times.csv").exists()


def test_cube_assembler_angle_offsets_applied(tmp_path, minimal_sphere_header):
    """SPHERE PA + true-north offsets are applied modulo 360."""
    config = _make_config(tmp_path)

    f1 = _make_science_file(
        tmp_path, "sci_1.fits", 1, minimal_sphere_header,
        angstart=0.0, angend=0.0, frame_size=63,
    )
    out_cube = tmp_path / "cube.fits"
    out_angs = tmp_path / "angs.txt"
    star_center = tmp_path / "centers.csv"

    CubeAssembler(
        config=config,
        science_files=[f1],
        output_file_cube=out_cube,
        output_file_angs=out_angs,
        shift=(0.0, 0.0),
        star_center_file=star_center,
    ).run()

    angs = np.loadtxt(out_angs)
    # angstart=angend=0 -> per-frame angle = 0 + 0.5*0 = 0; then add
    # SPHERE_PA_OFFSET (135.99) and true-north (-1.75); modulo 360.
    expected = (0.0 + 135.99 + (-1.75)) % 360
    assert float(angs) == pytest.approx(expected, rel=1e-6)


def test_cube_assembler_do_cubes_false_writes_only_angles(
    tmp_path, minimal_sphere_header
):
    """When do_cubes=False the cube FITS is NOT written; angles file is."""
    config = _make_config(tmp_path, do_cubes=False)

    f1 = _make_science_file(
        tmp_path, "sci_1.fits", 1, minimal_sphere_header, frame_size=63,
    )
    out_cube = tmp_path / "cube.fits"
    out_angs = tmp_path / "angs.txt"
    star_center = tmp_path / "centers.csv"

    CubeAssembler(
        config=config,
        science_files=[f1],
        output_file_cube=out_cube,
        output_file_angs=out_angs,
        shift=(0.0, 0.0),
        star_center_file=star_center,
    ).run()

    assert out_angs.exists()
    assert not out_cube.exists()


def test_cube_assembler_dither_csv_metadata(tmp_path, minimal_sphere_header):
    """Per-file dither + temp + exptime values are recorded in header_infos.csv."""
    config = _make_config(tmp_path)

    f1 = _make_science_file(
        tmp_path, "sci_1.fits", 2, minimal_sphere_header,
        dith_x=3.0, dith_y=-2.0, temp=292.5, exptime=2.5,
        frame_size=63,
    )
    out_cube = tmp_path / "cube.fits"
    out_angs = tmp_path / "angs.txt"
    star_center = tmp_path / "centers.csv"

    CubeAssembler(
        config=config,
        science_files=[f1],
        output_file_cube=out_cube,
        output_file_angs=out_angs,
        shift=(0.0, 0.0),
        star_center_file=star_center,
    ).run()

    df = pd.read_csv(tmp_path / "header_infos.csv")
    assert len(df) == 1
    row = df.iloc[0]
    assert row["file"] == "sci_1.fits"
    assert row["dither_x"] == pytest.approx(3.0)
    assert row["dither_y"] == pytest.approx(-2.0)
    assert row["filter_temp"] == pytest.approx(292.5)
    assert row["exp_time"] == pytest.approx(2.5)
    assert int(row["n_frames"]) == 2


def test_cube_assembler_missing_input_file(tmp_path):
    """Missing science file -> StepExecutionError."""
    config = _make_config(tmp_path)
    out_cube = tmp_path / "cube.fits"
    out_angs = tmp_path / "angs.txt"
    star_center = tmp_path / "centers.csv"

    step = CubeAssembler(
        config=config,
        science_files=[tmp_path / "ghost.fits"],
        output_file_cube=out_cube,
        output_file_angs=out_angs,
        shift=(0.0, 0.0),
        star_center_file=star_center,
    )
    with pytest.raises(StepExecutionError):
        step.run()


def test_cube_assembler_missing_header_keyword(
    tmp_path, minimal_sphere_header
):
    """Science file missing a required keyword -> HeaderKeyError."""
    config = _make_config(tmp_path)

    bad_header = minimal_sphere_header.copy()
    del bad_header["HIERARCH ESO TEL PARANG END"]
    cube = np.zeros((1, 64, 64), dtype=np.float32)
    f1 = tmp_path / "sci_bad.fits"
    fits.writeto(f1, cube, header=bad_header, overwrite=True)

    step = CubeAssembler(
        config=config,
        science_files=[f1],
        output_file_cube=tmp_path / "cube.fits",
        output_file_angs=tmp_path / "angs.txt",
        shift=(0.0, 0.0),
        star_center_file=tmp_path / "centers.csv",
    )
    with pytest.raises(HeaderKeyError):
        step.run()


def test_cube_assembler_resumability(tmp_path, minimal_sphere_header):
    """Pre-existing cube + overwrite=False -> file untouched."""
    config = _make_config(tmp_path, overwrite=False)

    f1 = _make_science_file(
        tmp_path, "sci_1.fits", 1, minimal_sphere_header, frame_size=63,
    )
    out_cube = tmp_path / "cube_existing.fits"
    out_angs = tmp_path / "angs.txt"
    star_center = tmp_path / "centers.csv"

    sentinel = np.full((1, 8, 8), 99.0, dtype=np.float32)
    fits.writeto(out_cube, sentinel, overwrite=True)

    CubeAssembler(
        config=config,
        science_files=[f1],
        output_file_cube=out_cube,
        output_file_angs=out_angs,
        shift=(0.0, 0.0),
        star_center_file=star_center,
    ).run()

    assert np.allclose(fits.getdata(out_cube), sentinel)


def test_cube_assembler_empty_input_raises():
    """Empty science_files -> InvalidConfigError on construction."""
    paths = PathConfig(raw_input_dir="/tmp", reduced_output_dir="/tmp")
    config = ReacherConfig(paths=paths)
    with pytest.raises(InvalidConfigError):
        CubeAssembler(
            config=config,
            science_files=[],
            output_file_cube="/tmp/cube.fits",
            output_file_angs="/tmp/angs.txt",
            shift=(0.0, 0.0),
            star_center_file="/tmp/centers.csv",
        )


def test_cube_assembler_180_crossing_correction(
    tmp_path, minimal_sphere_header
):
    """Angles spanning the +/-180 line are wrapped via the legacy logic."""
    config = _make_config(tmp_path)

    # angstart positive, angend negative, |start|+|end| > 180 -> wrap.
    f1 = _make_science_file(
        tmp_path, "sci_1.fits", 1, minimal_sphere_header,
        angstart=170.0, angend=-170.0, frame_size=63,
    )
    out_cube = tmp_path / "cube.fits"
    out_angs = tmp_path / "angs.txt"
    star_center = tmp_path / "centers.csv"

    CubeAssembler(
        config=config,
        science_files=[f1],
        output_file_cube=out_cube,
        output_file_angs=out_angs,
        shift=(0.0, 0.0),
        star_center_file=star_center,
    ).run()

    angs = np.loadtxt(out_angs)
    # angstart=170, angend=190 (after wrap); per-frame midpoint: 180.
    # plus offsets, mod 360.
    expected = (180.0 + 135.99 - 1.75) % 360
    assert float(angs) == pytest.approx(expected, rel=1e-4)


def test_cube_assembler_with_master_sky(tmp_path, minimal_sphere_header):
    """Master sky path triggers temperature + exposure-scaled subtraction."""
    config = _make_config(tmp_path)

    # Frame value 100. Sky value 50. Scale 1 (matching temp/exptime).
    # After subtraction the output (within nan_mask) should be 50.
    f1 = _make_science_file(
        tmp_path, "sci_1.fits", 1, minimal_sphere_header,
        temp=290.0, exptime=1.0, frame_size=63, fill_value=100.0,
    )

    sky = np.full((63, 63), 50.0, dtype=np.float32)
    sky_header = minimal_sphere_header.copy()
    sky_header["EXPTIME"] = 1.0
    sky_header["MEANTEMP"] = 290.0
    sky_path = tmp_path / "master_sky.fits"
    fits.writeto(sky_path, sky, header=sky_header, overwrite=True)

    out_cube = tmp_path / "cube.fits"
    out_angs = tmp_path / "angs.txt"
    star_center = tmp_path / "centers.csv"

    CubeAssembler(
        config=config,
        science_files=[f1],
        output_file_cube=out_cube,
        output_file_angs=out_angs,
        shift=(0.0, 0.0),
        star_center_file=star_center,
        master_sky_path=sky_path,
    ).run()

    cube = fits.getdata(out_cube)
    # Inside the original frame box (centered) the value should be ~50;
    # the padded margins are NaN.
    # 63x63 frame is centered at index (481..543)
    # NaN can creep in around the edges due to sub-pixel shift; pick the
    # central-most pixel which is unaffected.
    center_value = cube[0, 512, 512]
    assert center_value == pytest.approx(50.0, abs=1.0)

    hdr = fits.getheader(out_cube)
    assert hdr["MASTSKY"] == sky_path.name
    assert hdr["MNSKYTMP"] == pytest.approx(290.0)
