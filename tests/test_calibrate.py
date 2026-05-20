import numpy as np
import pytest
from astropy.io import fits

from REACHER.config import PathConfig, ProcessingConfig, ReacherConfig
from REACHER.exceptions import StepExecutionError
from REACHER.steps.calibrate import FileCalibrator


def _make_config(tmp_path, overwrite):
    raw_dir = tmp_path / "raw"
    reduced_dir = tmp_path / "reduced"
    raw_dir.mkdir(parents=True, exist_ok=True)
    reduced_dir.mkdir(parents=True, exist_ok=True)
    paths = PathConfig(raw_input_dir=str(raw_dir), reduced_output_dir=str(reduced_dir))
    processing = ProcessingConfig(overwrite=overwrite)
    return ReacherConfig(paths=paths, processing=processing)


def _write_calib_trio(tmp_path, science, dark, flat, header, prefix="cal"):
    """Write science + master dark + master flat FITS files into tmp_path."""
    sci_path = tmp_path / f"{prefix}_science.fits"
    dark_path = tmp_path / f"{prefix}_master_dark.fits"
    flat_path = tmp_path / f"{prefix}_master_flat.fits"
    fits.writeto(sci_path, science.astype(np.float32), header=header, overwrite=True)
    fits.writeto(dark_path, dark.astype(np.float32), header=header, overwrite=True)
    fits.writeto(flat_path, flat.astype(np.float32), header=header, overwrite=True)
    return sci_path, dark_path, flat_path


def test_file_calibrator_2d_happy_path(tmp_path, minimal_sphere_header):
    """2D science: (9 - 1) / 2 == 4 across the frame; provenance written."""
    config = _make_config(tmp_path, overwrite=True)

    science = np.full((16, 16), 9.0, dtype=np.float32)
    dark = np.full((16, 16), 1.0, dtype=np.float32)
    flat = np.full((16, 16), 2.0, dtype=np.float32)

    sci_path, dark_path, flat_path = _write_calib_trio(
        tmp_path, science, dark, flat, minimal_sphere_header, prefix="2d"
    )

    output_path = tmp_path / "calibrated_2d.fits"
    step = FileCalibrator(
        config=config,
        input_file=sci_path,
        output_file=output_path,
        master_dark_path=dark_path,
        master_flat_path=flat_path,
    )
    step.run()

    result = fits.getdata(output_path)
    assert result.shape == (16, 16)
    assert np.allclose(result, 4.0)

    header = fits.getheader(output_path)
    histories = [str(h).strip() for h in header.get("HISTORY", [])]
    # HISTORY1/2/3 are stored as keyword cards (as in the legacy routine).
    assert header.get("HISTORY1") == "DARK CORRECTED"
    assert header.get("HISTORY2") == "FLAT CORRECTED"
    assert header.get("HISTORY3") == "BAD PIXEL CORRECTED"
    assert header.get("MASTDARK") == dark_path.name
    assert header.get("MASTFLAT") == flat_path.name
    # BADPIX_FILE only set when a bad pixel mask was supplied.
    assert "BADPIX_FILE" not in header
    # Sanitized header passes through OBJECT (with spaces -> underscores).
    assert header.get("OBJECT") == "TEST"
    # No accidental HISTORY entries.
    assert histories == []


def test_file_calibrator_3d_happy_path(tmp_path, minimal_sphere_header):
    """3D science cube broadcasts dark/flat per frame; (9-1)/2 == 4."""
    config = _make_config(tmp_path, overwrite=True)

    science_cube = np.full((4, 16, 16), 9.0, dtype=np.float32)
    dark = np.full((16, 16), 1.0, dtype=np.float32)
    flat = np.full((16, 16), 2.0, dtype=np.float32)

    sci_path, dark_path, flat_path = _write_calib_trio(
        tmp_path, science_cube, dark, flat, minimal_sphere_header, prefix="3d"
    )

    output_path = tmp_path / "calibrated_3d.fits"
    FileCalibrator(
        config=config,
        input_file=sci_path,
        output_file=output_path,
        master_dark_path=dark_path,
        master_flat_path=flat_path,
    ).run()

    result = fits.getdata(output_path)
    assert result.shape == (4, 16, 16)
    assert np.allclose(result, 4.0)


def test_file_calibrator_bad_pixel_correction(tmp_path, minimal_sphere_header):
    """Hot pixel in science is interpolated to its 3x3 neighborhood median."""
    config = _make_config(tmp_path, overwrite=True)

    # Science = 9 everywhere except one hot pixel at (4, 4).
    science = np.full((9, 9), 9.0, dtype=np.float32)
    science[4, 4] = 9000.0
    dark = np.full((9, 9), 1.0, dtype=np.float32)
    flat = np.full((9, 9), 2.0, dtype=np.float32)

    sci_path, dark_path, flat_path = _write_calib_trio(
        tmp_path, science, dark, flat, minimal_sphere_header, prefix="bp"
    )

    bad_pix_mask = np.zeros((9, 9), dtype=np.float32)
    bad_pix_mask[4, 4] = 1.0
    bpm_path = tmp_path / "bpm.fits"
    fits.writeto(bpm_path, bad_pix_mask, overwrite=True)

    output_path = tmp_path / "calibrated_bp.fits"
    FileCalibrator(
        config=config,
        input_file=sci_path,
        output_file=output_path,
        master_dark_path=dark_path,
        master_flat_path=flat_path,
        bad_pix_mask_path=bpm_path,
    ).run()

    result = fits.getdata(output_path)
    # All non-flagged pixels are 4.0; the flagged pixel is interpolated to
    # the median of its 3x3 neighborhood (which is also 4.0 because the
    # surrounding science values are all 9 -> calibrated 4).
    assert result[0, 0] == pytest.approx(4.0)
    assert result[4, 4] == pytest.approx(4.0)
    # Sanity check: without the mask the flagged pixel would have been
    # (9000 - 1) / 2 == 4499.5
    assert result[4, 4] != pytest.approx(4499.5)

    header = fits.getheader(output_path)
    assert header.get("BADPIX_FILE") == bpm_path.name


def test_file_calibrator_missing_input(tmp_path, minimal_sphere_header):
    """Missing science file -> StepExecutionError."""
    config = _make_config(tmp_path, overwrite=True)

    dark = np.full((4, 4), 1.0, dtype=np.float32)
    flat = np.full((4, 4), 2.0, dtype=np.float32)
    dark_path = tmp_path / "dark.fits"
    flat_path = tmp_path / "flat.fits"
    fits.writeto(dark_path, dark, header=minimal_sphere_header, overwrite=True)
    fits.writeto(flat_path, flat, header=minimal_sphere_header, overwrite=True)

    output_path = tmp_path / "should_not_be_written.fits"
    step = FileCalibrator(
        config=config,
        input_file=tmp_path / "does_not_exist.fits",
        output_file=output_path,
        master_dark_path=dark_path,
        master_flat_path=flat_path,
    )

    with pytest.raises(StepExecutionError):
        step.run()
    assert not output_path.exists()


def test_file_calibrator_resumability(tmp_path, minimal_sphere_header):
    """Pre-existing output + overwrite=False -> file untouched."""
    config = _make_config(tmp_path, overwrite=False)

    science = np.full((8, 8), 9.0, dtype=np.float32)
    dark = np.full((8, 8), 1.0, dtype=np.float32)
    flat = np.full((8, 8), 2.0, dtype=np.float32)

    sci_path, dark_path, flat_path = _write_calib_trio(
        tmp_path, science, dark, flat, minimal_sphere_header, prefix="resume"
    )

    output_path = tmp_path / "calibrated_existing.fits"
    sentinel = np.full((8, 8), 99.0, dtype=np.float32)
    fits.writeto(output_path, sentinel, overwrite=True)

    FileCalibrator(
        config=config,
        input_file=sci_path,
        output_file=output_path,
        master_dark_path=dark_path,
        master_flat_path=flat_path,
    ).run()

    assert np.allclose(fits.getdata(output_path), sentinel)


def test_file_calibrator_master_background_not_implemented(
    tmp_path, minimal_sphere_header
):
    """Supplying master_background_path raises NotImplementedError.

    The legacy routine silently ignored this branch; we surface that
    instead of dropping user input.
    """
    config = _make_config(tmp_path, overwrite=True)

    science = np.full((4, 4), 9.0, dtype=np.float32)
    dark = np.full((4, 4), 1.0, dtype=np.float32)
    flat = np.full((4, 4), 2.0, dtype=np.float32)

    sci_path, dark_path, flat_path = _write_calib_trio(
        tmp_path, science, dark, flat, minimal_sphere_header, prefix="bkg"
    )

    bkg_path = tmp_path / "master_background.fits"
    fits.writeto(bkg_path, dark, overwrite=True)

    output_path = tmp_path / "calibrated_bkg.fits"
    step = FileCalibrator(
        config=config,
        input_file=sci_path,
        output_file=output_path,
        master_dark_path=dark_path,
        master_flat_path=flat_path,
        master_background_path=bkg_path,
    )

    with pytest.raises(NotImplementedError):
        step.run()
    assert not output_path.exists()
