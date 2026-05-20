import numpy as np
import pytest
from astropy.io import fits

from REACHER.config import PathConfig, ProcessingConfig, ReacherConfig
from REACHER.exceptions import StepExecutionError
from REACHER.steps.dark import MasterDarkCreator


def _make_config(tmp_path, overwrite):
    raw_dir = tmp_path / "raw"
    reduced_dir = tmp_path / "reduced"
    raw_dir.mkdir(parents=True, exist_ok=True)
    reduced_dir.mkdir(parents=True, exist_ok=True)
    paths = PathConfig(raw_input_dir=str(raw_dir), reduced_output_dir=str(reduced_dir))
    processing = ProcessingConfig(overwrite=overwrite)
    return ReacherConfig(paths=paths, processing=processing)


def test_master_dark_2d_inputs(tmp_path, minimal_sphere_header):
    """Happy path: two 2D darks median combine to expected per-pixel median."""
    config = _make_config(tmp_path, overwrite=True)

    dark_a = np.full((16, 16), 10.0, dtype=np.float32)
    dark_b = np.full((16, 16), 20.0, dtype=np.float32)

    path_a = tmp_path / "dark_a.fits"
    path_b = tmp_path / "dark_b.fits"
    fits.writeto(path_a, dark_a, header=minimal_sphere_header, overwrite=True)
    fits.writeto(path_b, dark_b, header=minimal_sphere_header, overwrite=True)

    master_dark_path = tmp_path / "master_dark_20200101.fits"

    step = MasterDarkCreator(
        config=config,
        dark_files=[path_a, path_b],
        master_dark_path=master_dark_path,
    )
    step.run()

    result = fits.getdata(master_dark_path)
    expected = np.median(np.stack([dark_a, dark_b]), axis=0)
    assert result.shape == (16, 16)
    assert np.allclose(result, expected)

    header = fits.getheader(master_dark_path)
    assert header.get("DARK0") == path_a.name
    assert header.get("DARK1") == path_b.name
    assert "MASTER DARK" in [str(h).strip() for h in header.get("HISTORY", [])]


def test_master_dark_3d_collapse(tmp_path, minimal_sphere_header):
    """3D cube input is collapsed to its nanmedian along axis 0 first."""
    config = _make_config(tmp_path, overwrite=True)

    cube = np.stack(
        [
            np.full((8, 8), 5.0, dtype=np.float32),
            np.full((8, 8), 7.0, dtype=np.float32),
            np.full((8, 8), 9.0, dtype=np.float32),
        ]
    )
    expected = np.nanmedian(cube, axis=0)  # constant 7.0

    dark_path = tmp_path / "dark_cube.fits"
    fits.writeto(dark_path, cube, header=minimal_sphere_header, overwrite=True)

    master_dark_path = tmp_path / "master_dark_3d.fits"

    step = MasterDarkCreator(
        config=config,
        dark_files=[dark_path],
        master_dark_path=master_dark_path,
    )
    step.run()

    result = fits.getdata(master_dark_path)
    assert result.shape == (8, 8)
    assert np.allclose(result, expected)


def test_master_dark_missing_input(tmp_path):
    """Missing dark file -> StepExecutionError."""
    config = _make_config(tmp_path, overwrite=True)
    master_dark_path = tmp_path / "master_dark.fits"

    step = MasterDarkCreator(
        config=config,
        dark_files=[tmp_path / "does_not_exist.fits"],
        master_dark_path=master_dark_path,
    )

    with pytest.raises(StepExecutionError):
        step.run()


def test_master_dark_resumability(tmp_path, minimal_sphere_header):
    """Pre-existing master dark + overwrite=False -> file is left untouched."""
    config = _make_config(tmp_path, overwrite=False)

    dark_path = tmp_path / "dark.fits"
    fits.writeto(
        dark_path,
        np.full((8, 8), 42.0, dtype=np.float32),
        header=minimal_sphere_header,
        overwrite=True,
    )

    master_dark_path = tmp_path / "master_dark_existing.fits"
    sentinel = np.full((8, 8), 99.0, dtype=np.float32)
    fits.writeto(master_dark_path, sentinel, overwrite=True)

    step = MasterDarkCreator(
        config=config,
        dark_files=[dark_path],
        master_dark_path=master_dark_path,
    )
    step.run()

    assert np.allclose(fits.getdata(master_dark_path), sentinel)


def test_master_dark_bad_pixel_correction(tmp_path, minimal_sphere_header):
    """Supplying a bad pixel mask interpolates the affected pixel."""
    config = _make_config(tmp_path, overwrite=True)

    # Two darks both with the same hot pixel injected at (4, 4).
    base = np.ones((9, 9), dtype=np.float32)
    dark_a = base.copy()
    dark_b = base.copy()
    dark_a[4, 4] = 1000.0
    dark_b[4, 4] = 1000.0

    path_a = tmp_path / "dark_a.fits"
    path_b = tmp_path / "dark_b.fits"
    fits.writeto(path_a, dark_a, header=minimal_sphere_header, overwrite=True)
    fits.writeto(path_b, dark_b, header=minimal_sphere_header, overwrite=True)

    bad_pix_mask = np.zeros((9, 9), dtype=np.float32)
    bad_pix_mask[4, 4] = 1.0
    bad_pix_mask_path = tmp_path / "bpm.fits"
    fits.writeto(bad_pix_mask_path, bad_pix_mask, overwrite=True)

    master_dark_path = tmp_path / "master_dark_bp.fits"

    step = MasterDarkCreator(
        config=config,
        dark_files=[path_a, path_b],
        master_dark_path=master_dark_path,
        bad_pix_mask_path=bad_pix_mask_path,
    )
    step.run()

    result = fits.getdata(master_dark_path)
    # Without correction the pixel would be 1000.0; after 3x3 median of
    # neighbors (all 1.0) it should be 1.0.
    assert result[4, 4] == pytest.approx(1.0)
    # Surrounding pixels untouched.
    assert result[0, 0] == pytest.approx(1.0)
