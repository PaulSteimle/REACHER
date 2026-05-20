import numpy as np
import pytest
from astropy.io import fits

from REACHER.config import PathConfig, ProcessingConfig, ReacherConfig
from REACHER.exceptions import StepExecutionError
from REACHER.steps.bad_pixel import BadPixelMaskCreator, interpolate_bad_pixels


def _make_config(tmp_path, overwrite):
    raw_dir = tmp_path / "raw"
    reduced_dir = tmp_path / "reduced"
    raw_dir.mkdir(parents=True, exist_ok=True)
    reduced_dir.mkdir(parents=True, exist_ok=True)
    paths = PathConfig(raw_input_dir=str(raw_dir), reduced_output_dir=str(reduced_dir))
    processing = ProcessingConfig(overwrite=overwrite)
    return ReacherConfig(paths=paths, processing=processing)


def test_bad_pixel_mask_creation(tmp_path, synthetic_dark_cube, minimal_sphere_header):
    config = _make_config(tmp_path, overwrite=True)
    dark_cube, hot_pixels, cold_pixels = synthetic_dark_cube

    dark_path = tmp_path / "dark_20200101.fits"
    fits.writeto(dark_path, dark_cube, header=minimal_sphere_header, overwrite=True)

    bad_mask_path = tmp_path / "bad_pix_mask_20200101.fits"
    clump_mask_path = tmp_path / "IRDIS_clump_mask.fits"

    step = BadPixelMaskCreator(
        config=config,
        dark_files=[dark_path],
        bad_pix_mask_path=bad_mask_path,
        clump_mask_path=clump_mask_path,
        coef=5.0,
    )
    step.run()

    mask = fits.getdata(bad_mask_path)
    expected_count = len(hot_pixels) + len(cold_pixels)
    assert int(mask.sum()) == expected_count


def test_interpolate_bad_pixels():
    image = np.ones((5, 5), dtype=float)
    image[2, 2] = 100.0
    mask = np.zeros_like(image)
    mask[2, 2] = 1.0

    corrected = interpolate_bad_pixels(image, mask)
    assert corrected[2, 2] == 1.0


def test_bad_pixel_missing_dark(tmp_path):
    config = _make_config(tmp_path, overwrite=True)
    bad_mask_path = tmp_path / "bad_pix_mask_20200101.fits"
    clump_mask_path = tmp_path / "IRDIS_clump_mask.fits"

    step = BadPixelMaskCreator(
        config=config,
        dark_files=[tmp_path / "missing.fits"],
        bad_pix_mask_path=bad_mask_path,
        clump_mask_path=clump_mask_path,
    )

    with pytest.raises(StepExecutionError):
        step.run()


def test_bad_pixel_resumability(tmp_path, synthetic_dark_cube, minimal_sphere_header):
    config = _make_config(tmp_path, overwrite=False)
    dark_cube, _, _ = synthetic_dark_cube

    dark_path = tmp_path / "dark_20200101.fits"
    fits.writeto(dark_path, dark_cube, header=minimal_sphere_header, overwrite=True)

    bad_mask_path = tmp_path / "bad_pix_mask_20200101.fits"
    clump_mask_path = tmp_path / "IRDIS_clump_mask.fits"

    existing_bad_mask = np.ones((1024, 2048), dtype=np.float32) * 2.0
    existing_clump_mask = np.ones((1024, 2048), dtype=np.float32) * 3.0
    fits.writeto(bad_mask_path, existing_bad_mask, overwrite=True)
    fits.writeto(clump_mask_path, existing_clump_mask, overwrite=True)

    step = BadPixelMaskCreator(
        config=config,
        dark_files=[dark_path],
        bad_pix_mask_path=bad_mask_path,
        clump_mask_path=clump_mask_path,
    )
    step.run()

    assert np.allclose(fits.getdata(bad_mask_path), existing_bad_mask)
    assert np.allclose(fits.getdata(clump_mask_path), existing_clump_mask)
