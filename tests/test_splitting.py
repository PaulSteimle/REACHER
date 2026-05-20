"""Tests for ChannelSplitter step."""
import numpy as np
import pytest
from astropy.io import fits

from REACHER.config import PathConfig, ProcessingConfig, ReacherConfig
from REACHER.exceptions import ShapeMismatchError, StepExecutionError
from REACHER.IRDIS_instrument_constants import (
    IRDIS_LEFT_CHANNEL,
    IRDIS_RIGHT_CHANNEL,
)
from REACHER.steps.splitting import ChannelSplitter


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


def test_channel_splitter_happy_path(tmp_path, minimal_sphere_header):
    """Full IRDIS-shaped cube splits to two (N, 995, 875) outputs."""
    config = _make_config(tmp_path, overwrite=True)

    n_frames = 3
    cube = np.full((n_frames, 1024, 2048), 7.0, dtype=np.float32)
    in_path = tmp_path / "science.fits"
    fits.writeto(in_path, cube, header=minimal_sphere_header, overwrite=True)

    left_path = tmp_path / "left.fits"
    right_path = tmp_path / "right.fits"

    ChannelSplitter(
        config=config,
        input_file=in_path,
        left_output_file=left_path,
        right_output_file=right_path,
    ).run()

    left_data = fits.getdata(left_path)
    right_data = fits.getdata(right_path)

    assert left_data.shape == (n_frames, 995, 875)
    assert right_data.shape == (n_frames, 995, 875)
    assert np.allclose(left_data, 7.0)
    assert np.allclose(right_data, 7.0)

    # Header passes through with HIERARCH keywords intact.
    left_hdr = fits.getheader(left_path)
    assert left_hdr.get("OBJECT") == "TEST"
    assert left_hdr.get("HIERARCH ESO INS COMB IFLT") == "H"


def test_channel_splitter_channel_content(tmp_path, minimal_sphere_header):
    """LEFT and RIGHT slices pull from the right detector regions."""
    config = _make_config(tmp_path, overwrite=True)

    cube = np.zeros((2, 1024, 2048), dtype=np.float32)
    rows_l, cols_l = (
        IRDIS_LEFT_CHANNEL["rows"],
        IRDIS_LEFT_CHANNEL["cols"],
    )
    rows_r, cols_r = (
        IRDIS_RIGHT_CHANNEL["rows"],
        IRDIS_RIGHT_CHANNEL["cols"],
    )
    # Distinct sentinel values for LEFT vs RIGHT detector halves.
    cube[:, rows_l, cols_l] = 11.0
    cube[:, rows_r, cols_r] = 22.0

    in_path = tmp_path / "science.fits"
    fits.writeto(in_path, cube, header=minimal_sphere_header, overwrite=True)

    left_path = tmp_path / "left.fits"
    right_path = tmp_path / "right.fits"

    ChannelSplitter(
        config=config,
        input_file=in_path,
        left_output_file=left_path,
        right_output_file=right_path,
    ).run()

    left_data = fits.getdata(left_path)
    right_data = fits.getdata(right_path)

    assert np.allclose(left_data, 11.0)
    assert np.allclose(right_data, 22.0)
    # Sanity: LEFT and RIGHT outputs are not identical.
    assert not np.allclose(left_data, right_data)


def test_channel_splitter_missing_input(tmp_path):
    """Missing input file -> StepExecutionError, no outputs written."""
    config = _make_config(tmp_path, overwrite=True)

    left_path = tmp_path / "left.fits"
    right_path = tmp_path / "right.fits"

    step = ChannelSplitter(
        config=config,
        input_file=tmp_path / "does_not_exist.fits",
        left_output_file=left_path,
        right_output_file=right_path,
    )

    with pytest.raises(StepExecutionError):
        step.run()
    assert not left_path.exists()
    assert not right_path.exists()


def test_channel_splitter_2d_input_raises(tmp_path, minimal_sphere_header):
    """A 2D image is rejected explicitly (legacy code would IndexError)."""
    config = _make_config(tmp_path, overwrite=True)

    flat_image = np.zeros((1024, 2048), dtype=np.float32)
    in_path = tmp_path / "science_2d.fits"
    fits.writeto(
        in_path, flat_image, header=minimal_sphere_header, overwrite=True
    )

    left_path = tmp_path / "left.fits"
    right_path = tmp_path / "right.fits"

    step = ChannelSplitter(
        config=config,
        input_file=in_path,
        left_output_file=left_path,
        right_output_file=right_path,
    )

    with pytest.raises(ShapeMismatchError):
        step.run()
    assert not left_path.exists()
    assert not right_path.exists()


def test_channel_splitter_resumability_both_exist(
    tmp_path, minimal_sphere_header
):
    """Both LEFT and RIGHT exist + overwrite=False -> files left untouched."""
    config = _make_config(tmp_path, overwrite=False)

    cube = np.full((2, 1024, 2048), 3.0, dtype=np.float32)
    in_path = tmp_path / "science.fits"
    fits.writeto(in_path, cube, header=minimal_sphere_header, overwrite=True)

    left_path = tmp_path / "left.fits"
    right_path = tmp_path / "right.fits"
    sentinel_left = np.full((2, 995, 875), 91.0, dtype=np.float32)
    sentinel_right = np.full((2, 995, 875), 92.0, dtype=np.float32)
    fits.writeto(left_path, sentinel_left, overwrite=True)
    fits.writeto(right_path, sentinel_right, overwrite=True)

    ChannelSplitter(
        config=config,
        input_file=in_path,
        left_output_file=left_path,
        right_output_file=right_path,
    ).run()

    assert np.allclose(fits.getdata(left_path), 91.0)
    assert np.allclose(fits.getdata(right_path), 92.0)


def test_channel_splitter_resumability_partial(tmp_path, minimal_sphere_header):
    """Only LEFT exists + overwrite=False -> step still runs, both rewritten.

    Mirrors legacy behavior: the skip condition requires BOTH outputs to
    exist; otherwise the splitter runs and overwrites both files.
    """
    config = _make_config(tmp_path, overwrite=False)

    cube = np.full((1, 1024, 2048), 5.0, dtype=np.float32)
    in_path = tmp_path / "science.fits"
    fits.writeto(in_path, cube, header=minimal_sphere_header, overwrite=True)

    left_path = tmp_path / "left.fits"
    right_path = tmp_path / "right.fits"
    sentinel_left = np.full((1, 995, 875), 91.0, dtype=np.float32)
    fits.writeto(left_path, sentinel_left, overwrite=True)
    # right_path intentionally missing.

    ChannelSplitter(
        config=config,
        input_file=in_path,
        left_output_file=left_path,
        right_output_file=right_path,
    ).run()

    # Both outputs now reflect the input cube value.
    assert np.allclose(fits.getdata(left_path), 5.0)
    assert np.allclose(fits.getdata(right_path), 5.0)
