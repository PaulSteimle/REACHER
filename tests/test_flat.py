import numpy as np
import pytest
from astropy.io import fits

from REACHER.config import PathConfig, ProcessingConfig, ReacherConfig
from REACHER.exceptions import StepExecutionError
from REACHER.steps.flat import MasterFlatCreator


def _make_config(tmp_path, overwrite):
    raw_dir = tmp_path / "raw"
    reduced_dir = tmp_path / "reduced"
    raw_dir.mkdir(parents=True, exist_ok=True)
    reduced_dir.mkdir(parents=True, exist_ok=True)
    paths = PathConfig(raw_input_dir=str(raw_dir), reduced_output_dir=str(reduced_dir))
    processing = ProcessingConfig(overwrite=overwrite)
    return ReacherConfig(paths=paths, processing=processing)


def _write_full_irdis_flat(tmp_path, name, dark_value, gradient, header):
    """Write a synthetic IRDIS-shape flat (1024x2048) and matching dark."""
    flat = gradient + dark_value
    dark = np.full((1024, 2048), dark_value, dtype=np.float32)
    flat_path = tmp_path / f"flat_{name}.fits"
    dark_path = tmp_path / f"dark_{name}.fits"
    fits.writeto(flat_path, flat.astype(np.float32), header=header, overwrite=True)
    fits.writeto(dark_path, dark, header=header, overwrite=True)
    return flat_path, dark_path


def test_master_flat_happy_path(tmp_path, minimal_sphere_header, synthetic_flat):
    """Synthetic gradient flat collapses to ~1.0 after final normalization.

    Uses the full IRDIS frame shape so the LEFT/RIGHT channel slices match
    the layout in instrument_constants.py.
    """
    config = _make_config(tmp_path, overwrite=True)

    flat_path, dark_path = _write_full_irdis_flat(
        tmp_path, "a", dark_value=10.0, gradient=synthetic_flat, header=minimal_sphere_header
    )

    master_flat_path = tmp_path / "master_flat_happy.fits"
    step = MasterFlatCreator(
        config=config,
        flat_files=[flat_path],
        master_dark_files=[dark_path],
        master_flat_path=master_flat_path,
    )
    step.run()

    result = fits.getdata(master_flat_path)
    assert result.shape == (1024, 2048)
    # The algorithm's invariant: mean of LEFT and RIGHT channel medians == 1.0
    # (median of the full frame is not necessarily 1.0 because pixels outside
    # the LEFT/RIGHT slices are not part of the normalization).
    left_med = np.nanmedian(result[25:1020, 50:925])
    right_med = np.nanmedian(result[15:1010, 1080:1955])
    assert (left_med + right_med) / 2 == pytest.approx(1.0, abs=1e-6)
    # And the global median is close to (but not exactly) 1.0 for our
    # synthetic gradient.
    assert np.nanmedian(result) == pytest.approx(1.0, abs=5e-3)

    header = fits.getheader(master_flat_path)
    assert header.get("FLAT0") == flat_path.name
    assert "MASTER FLAT" in [str(h).strip() for h in header.get("HISTORY", [])]


def test_master_flat_per_frame_normalization(
    tmp_path, minimal_sphere_header, synthetic_flat
):
    """Two flats with different overall levels collapse to the same value.

    Per-frame normalization (divide by per-frame median) should make the
    final master flat invariant to a scalar multiplicative scaling of the
    inputs.
    """
    config = _make_config(tmp_path, overwrite=True)

    # Same gradient, but the second flat is scaled up by a factor of 5.
    flat_a = synthetic_flat
    flat_b = synthetic_flat * 5.0

    dark = np.full((1024, 2048), 10.0, dtype=np.float32)
    flat_a_path = tmp_path / "flat_a.fits"
    flat_b_path = tmp_path / "flat_b.fits"
    dark_path = tmp_path / "dark.fits"
    fits.writeto(
        flat_a_path,
        (flat_a + 10.0).astype(np.float32),
        header=minimal_sphere_header,
        overwrite=True,
    )
    fits.writeto(
        flat_b_path,
        (flat_b + 10.0).astype(np.float32),
        header=minimal_sphere_header,
        overwrite=True,
    )
    fits.writeto(dark_path, dark, header=minimal_sphere_header, overwrite=True)

    out_two = tmp_path / "master_flat_two.fits"
    MasterFlatCreator(
        config=config,
        flat_files=[flat_a_path, flat_b_path],
        master_dark_files=[dark_path, dark_path],
        master_flat_path=out_two,
    ).run()

    out_one = tmp_path / "master_flat_one.fits"
    MasterFlatCreator(
        config=config,
        flat_files=[flat_a_path],
        master_dark_files=[dark_path],
        master_flat_path=out_one,
    ).run()

    result_two = fits.getdata(out_two)
    result_one = fits.getdata(out_one)
    # Per-frame normalization makes the global scale of each flat irrelevant.
    assert np.allclose(result_two, result_one, atol=1e-5)


def test_master_flat_missing_input(tmp_path):
    """Missing flat file -> StepExecutionError."""
    config = _make_config(tmp_path, overwrite=True)
    master_flat_path = tmp_path / "master_flat_missing.fits"

    step = MasterFlatCreator(
        config=config,
        flat_files=[tmp_path / "does_not_exist.fits"],
        master_dark_files=[tmp_path / "also_missing.fits"],
        master_flat_path=master_flat_path,
    )

    with pytest.raises(StepExecutionError):
        step.run()


def test_master_flat_resumability(tmp_path, minimal_sphere_header, synthetic_flat):
    """Pre-existing master flat + overwrite=False -> file untouched."""
    config = _make_config(tmp_path, overwrite=False)

    flat_path, dark_path = _write_full_irdis_flat(
        tmp_path, "a", dark_value=10.0, gradient=synthetic_flat, header=minimal_sphere_header
    )

    master_flat_path = tmp_path / "master_flat_existing.fits"
    sentinel = np.full((4, 4), 99.0, dtype=np.float32)
    fits.writeto(master_flat_path, sentinel, overwrite=True)

    step = MasterFlatCreator(
        config=config,
        flat_files=[flat_path],
        master_dark_files=[dark_path],
        master_flat_path=master_flat_path,
    )
    step.run()

    assert np.allclose(fits.getdata(master_flat_path), sentinel)


def test_master_flat_left_right_channel_norm(
    tmp_path, minimal_sphere_header, synthetic_flat
):
    """LEFT and RIGHT channel medians independently combine to ~1.0.

    Inject a deliberate brightness offset between the two IRDIS channels
    and verify that the final-normalization step normalizes their two
    medians' mean to 1.0, i.e. that the medians of each channel
    individually average to 1.0.
    """
    config = _make_config(tmp_path, overwrite=True)

    # Build a flat where the LEFT channel is brighter than the RIGHT.
    # Use the full IRDIS shape so the channel slices apply.
    flat_full = synthetic_flat.copy()
    # Left bright, right faint via a mask in a column-wise band.
    flat_full[:, :1024] *= 1.5
    flat_full[:, 1024:] *= 0.5

    dark_value = 10.0
    flat_path, dark_path = _write_full_irdis_flat(
        tmp_path,
        "lr",
        dark_value=dark_value,
        gradient=flat_full,
        header=minimal_sphere_header,
    )

    master_flat_path = tmp_path / "master_flat_lr.fits"
    MasterFlatCreator(
        config=config,
        flat_files=[flat_path],
        master_dark_files=[dark_path],
        master_flat_path=master_flat_path,
    ).run()

    result = fits.getdata(master_flat_path)

    # LEFT/RIGHT slices straight from instrument_constants.
    left_med = np.nanmedian(result[25:1020, 50:925])
    right_med = np.nanmedian(result[15:1010, 1080:1955])

    # The mean of the two medians is exactly the normalization factor,
    # so it must be ~1.0.
    assert (left_med + right_med) / 2 == pytest.approx(1.0, abs=1e-6)
    # The two medians themselves must straddle 1.0 (since left was brighter
    # before normalization, its median should be >1 and right should be <1).
    assert left_med > 1.0
    assert right_med < 1.0
