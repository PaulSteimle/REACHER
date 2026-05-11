"""Tests for FluxCalibrator step and the ND-filter ratio utility."""
import numpy as np
import pytest
from astropy.io import fits

from REACHER.config import (
    CalibrationConfig,
    PathConfig,
    ProcessingConfig,
    ReacherConfig,
)
from REACHER.exceptions import (
    HeaderKeyError,
    InvalidConfigError,
    StepExecutionError,
)
from REACHER.steps.flux import FluxCalibrator
from REACHER.utils.filters import get_ratio_for_nd_filter


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------
def _make_config(
    tmp_path,
    overwrite=True,
    make_plots=False,
    star_loc=(40, 40),
    cut_out_size=21,
):
    raw_dir = tmp_path / "raw"
    reduced_dir = tmp_path / "reduced"
    raw_dir.mkdir(parents=True, exist_ok=True)
    reduced_dir.mkdir(parents=True, exist_ok=True)
    paths = PathConfig(
        raw_input_dir=str(raw_dir), reduced_output_dir=str(reduced_dir)
    )
    processing = ProcessingConfig(overwrite=overwrite, make_plots=make_plots)
    calibration = CalibrationConfig(
        star_loc=star_loc,
        flux_cut_out_size=cut_out_size,
    )
    return ReacherConfig(
        paths=paths, processing=processing, calibration=calibration
    )


def _gauss2d(shape, center, sigma=2.5, amplitude=1000.0):
    """Build a 2D Gaussian image."""
    yy, xx = np.indices(shape)
    cx, cy = center
    return amplitude * np.exp(
        -(((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
    )


def _flux_header(base_header, exp_time, wave_filter="BB_H", nd_filter="ND_2.0"):
    h = base_header.copy()
    h["EXPTIME"] = float(exp_time)
    h["HIERARCH ESO INS COMB IFLT"] = wave_filter
    h["HIERARCH ESO INS4 FILT2 ID"] = nd_filter
    h["HIERARCH ESO INS4 TEMP449 VAL"] = 290.0
    return h


# ---------------------------------------------------------------------------
# get_ratio_for_nd_filter
# ---------------------------------------------------------------------------
def test_nd_ratio_known_values():
    """ND ratio is between 0 and 1 and matches the legacy integration.

    Sanity checks for the BB_H + ND_2.0 ratio (ND=2.0 is ~10^-2 attenuation).
    """
    r = get_ratio_for_nd_filter("BB_H", "ND_2.0", "LEFT")
    assert 0.0 < r < 1.0
    # ND 2.0 attenuates the signal by a factor of ~100, so the ratio
    # should be well below 0.1.
    assert r < 0.1


def test_nd_ratio_db_filter_two_channels_differ():
    """DB filters have separate LEFT/RIGHT transmission columns."""
    rl = get_ratio_for_nd_filter("DB_H23", "ND_1.0", "LEFT")
    rr = get_ratio_for_nd_filter("DB_H23", "ND_1.0", "RIGHT")
    assert rl != rr
    assert 0.0 < rl < 1.0
    assert 0.0 < rr < 1.0


def test_nd_ratio_invalid_channel_raises():
    with pytest.raises(InvalidConfigError):
        get_ratio_for_nd_filter("DB_H23", "ND_1.0", "MIDDLE")


# ---------------------------------------------------------------------------
# FluxCalibrator
# ---------------------------------------------------------------------------
def test_flux_calibrator_happy_path(tmp_path, minimal_sphere_header):
    """End-to-end: produces a centered postage stamp with provenance keys."""
    config = _make_config(
        tmp_path, overwrite=True, make_plots=False,
        star_loc=(40, 40), cut_out_size=21,
    )

    # Build a 3-frame cube with a Gaussian PSF at (cx=40, cy=40).
    frame = _gauss2d((81, 81), center=(40, 40), sigma=2.5, amplitude=1000.0)
    cube = np.stack([frame] * 3).astype(np.float32)

    flux_path = tmp_path / "flux.fits"
    flux_header = _flux_header(minimal_sphere_header, exp_time=1.0)
    fits.writeto(flux_path, cube, header=flux_header, overwrite=True)

    output_path = tmp_path / "flux_calibrated.fits"
    FluxCalibrator(
        config=config,
        input_file=flux_path,
        output_file=output_path,
        science_exp_time=64.0,
        channel="LEFT",
    ).run()

    out_data = fits.getdata(output_path)
    out_header = fits.getheader(output_path)

    # Cut size 21 is odd -> legacy increment makes it 22 (even) -> the
    # stamp size is (cut_out_size-1)/2 floor + 1 along each axis.
    # cut_out_size becomes 22; cut_out_half = 10; stamp = 21x21.
    assert out_data.shape == (21, 21)

    # Provenance keywords.
    assert out_header["EXPTIMERATIO"] == pytest.approx(64.0)
    nd_ratio = get_ratio_for_nd_filter("BB_H", "ND_2.0", "LEFT")
    assert out_header["ND_RATIO"] == pytest.approx(1.0 / nd_ratio)

    # Scaling check: the peak of the stamp should be roughly
    # amplitude * exp_time_ratio / nd_ratio (minus background).
    expected_scale = 1000.0 * 64.0 / nd_ratio
    peak = np.nanmax(out_data)
    # Within 5% of the expected peak.
    assert peak == pytest.approx(expected_scale, rel=0.05)


def test_flux_calibrator_missing_input(tmp_path):
    """Missing input file -> StepExecutionError."""
    config = _make_config(tmp_path, overwrite=True, make_plots=False)
    output_path = tmp_path / "out.fits"
    step = FluxCalibrator(
        config=config,
        input_file=tmp_path / "ghost.fits",
        output_file=output_path,
        science_exp_time=10.0,
        channel="LEFT",
    )
    with pytest.raises(StepExecutionError):
        step.run()
    assert not output_path.exists()


def test_flux_calibrator_missing_header_keyword(
    tmp_path, minimal_sphere_header
):
    """Missing required header key -> HeaderKeyError."""
    config = _make_config(tmp_path, overwrite=True, make_plots=False)

    cube = np.zeros((2, 60, 60), dtype=np.float32)
    flux_path = tmp_path / "flux_bad.fits"
    bad_header = minimal_sphere_header.copy()
    # remove the wave filter keyword.
    del bad_header["HIERARCH ESO INS COMB IFLT"]
    bad_header["EXPTIME"] = 1.0
    bad_header["HIERARCH ESO INS4 TEMP449 VAL"] = 290.0
    bad_header["HIERARCH ESO INS4 FILT2 ID"] = "ND_2.0"
    fits.writeto(flux_path, cube, header=bad_header, overwrite=True)

    output_path = tmp_path / "out.fits"
    with pytest.raises(HeaderKeyError):
        FluxCalibrator(
            config=config,
            input_file=flux_path,
            output_file=output_path,
            science_exp_time=10.0,
            channel="LEFT",
        ).run()


def test_flux_calibrator_invalid_channel():
    """An invalid channel string is rejected at construction."""
    paths = PathConfig(raw_input_dir="/tmp", reduced_output_dir="/tmp")
    processing = ProcessingConfig()
    calibration = CalibrationConfig()
    config = ReacherConfig(
        paths=paths, processing=processing, calibration=calibration
    )
    with pytest.raises(InvalidConfigError):
        FluxCalibrator(
            config=config,
            input_file="/tmp/x.fits",
            output_file="/tmp/y.fits",
            science_exp_time=1.0,
            channel="MIDDLE",
        )


def test_flux_calibrator_resumability(tmp_path, minimal_sphere_header):
    """Pre-existing output + overwrite=False -> file untouched."""
    config = _make_config(tmp_path, overwrite=False, make_plots=False)

    # Need a real input file for _check_inputs.
    cube = np.zeros((2, 81, 81), dtype=np.float32)
    flux_path = tmp_path / "flux.fits"
    flux_header = _flux_header(minimal_sphere_header, exp_time=1.0)
    fits.writeto(flux_path, cube, header=flux_header, overwrite=True)

    output_path = tmp_path / "flux_existing.fits"
    sentinel = np.full((21, 21), 99.0, dtype=np.float32)
    fits.writeto(output_path, sentinel, overwrite=True)

    FluxCalibrator(
        config=config,
        input_file=flux_path,
        output_file=output_path,
        science_exp_time=10.0,
        channel="LEFT",
    ).run()

    assert np.allclose(fits.getdata(output_path), sentinel)


def test_flux_calibrator_nan_center_raises(tmp_path, minimal_sphere_header):
    """A blank cube produces a NaN Gaussian fit -> StepExecutionError.

    The legacy code referenced an undefined ``rescaled_image`` here
    causing a ``NameError``; the step now raises a clear, typed error.
    """
    config = _make_config(
        tmp_path, overwrite=True, make_plots=False,
        star_loc=(40, 40), cut_out_size=21,
    )

    # No signal whatsoever -> Gaussian fit cannot converge.
    cube = np.zeros((2, 81, 81), dtype=np.float32)
    flux_path = tmp_path / "flux_blank.fits"
    flux_header = _flux_header(minimal_sphere_header, exp_time=1.0)
    fits.writeto(flux_path, cube, header=flux_header, overwrite=True)

    output_path = tmp_path / "out_blank.fits"
    step = FluxCalibrator(
        config=config,
        input_file=flux_path,
        output_file=output_path,
        science_exp_time=10.0,
        channel="LEFT",
    )
    with pytest.raises(StepExecutionError):
        step.run()


def test_flux_calibrator_with_master_sky(tmp_path, minimal_sphere_header):
    """Master sky path triggers temperature/exposure-scaled subtraction."""
    config = _make_config(
        tmp_path, overwrite=True, make_plots=False,
        star_loc=(40, 40), cut_out_size=21,
    )

    # Flux cube: signal + DC offset of 50 counts.
    frame = _gauss2d((81, 81), center=(40, 40), sigma=2.5, amplitude=1000.0)
    cube = np.stack([frame + 50.0] * 3).astype(np.float32)

    flux_path = tmp_path / "flux.fits"
    flux_header = _flux_header(minimal_sphere_header, exp_time=1.0)
    fits.writeto(flux_path, cube, header=flux_header, overwrite=True)

    # Master sky: flat 50 counts; same temperature and exposure time
    # so the scaling factor is exactly 1.
    sky = np.full((81, 81), 50.0, dtype=np.float32)
    sky_header = minimal_sphere_header.copy()
    sky_header["EXPTIME"] = 1.0
    sky_header["MEANTEMP"] = 290.0
    sky_path = tmp_path / "master_sky.fits"
    fits.writeto(sky_path, sky, header=sky_header, overwrite=True)

    output_path = tmp_path / "flux_with_sky.fits"
    FluxCalibrator(
        config=config,
        input_file=flux_path,
        output_file=output_path,
        science_exp_time=64.0,
        channel="LEFT",
        master_sky_path=sky_path,
    ).run()

    out_header = fits.getheader(output_path)
    # Background should now be near zero (DC offset removed by sky sub).
    assert abs(out_header["BACKGROUND"]) < 1.0
