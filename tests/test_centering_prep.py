"""Tests for CenteringPreparer step."""
import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from REACHER.config import (
    PathConfig,
    ProcessingConfig,
    ReacherConfig,
)
from REACHER.exceptions import (
    InvalidConfigError,
    ShapeMismatchError,
    StepExecutionError,
)
from REACHER.steps.centering_prep import CenteringPreparer


def _make_config(tmp_path, overwrite=True, make_plots=False):
    raw_dir = tmp_path / "raw"
    reduced_dir = tmp_path / "reduced"
    raw_dir.mkdir(parents=True, exist_ok=True)
    reduced_dir.mkdir(parents=True, exist_ok=True)
    paths = PathConfig(
        raw_input_dir=str(raw_dir), reduced_output_dir=str(reduced_dir)
    )
    processing = ProcessingConfig(overwrite=overwrite, make_plots=make_plots)
    return ReacherConfig(paths=paths, processing=processing)


def _write_cube(tmp_path, name, n_frames, header):
    """Write a small 3D synthetic cube."""
    cube = np.zeros((n_frames, 64, 64), dtype=np.float32)
    # inject a faint Gaussian-ish blob near the center so the optional
    # centroid path has something sensible to find (only matters when
    # make_plots=True).
    cy, cx = 32, 32
    yy, xx = np.ogrid[:64, :64]
    cube[:] = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 5 ** 2)))
    fits_path = tmp_path / name
    fits.writeto(fits_path, cube, header=header, overwrite=True)
    return fits_path


def test_centering_prep_creates_csv(tmp_path, minimal_sphere_header):
    """Happy path: centers.csv created with placeholder rows for each file."""
    config = _make_config(tmp_path, overwrite=True, make_plots=False)
    output_dir = tmp_path / "out"

    f1 = _write_cube(tmp_path, "frame_001.fits", 3, minimal_sphere_header)
    f2 = _write_cube(tmp_path, "frame_002.fits", 3, minimal_sphere_header)

    CenteringPreparer(
        config=config,
        input_files=[f1, f2],
        output_dir=output_dir,
    ).run()

    csv_path = output_dir / "centers.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)

    expected_cols = [
        "frames", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4",
    ]
    assert list(df.columns) == expected_cols
    assert sorted(df["frames"].tolist()) == [
        "frame_001.fits", "frame_002.fits"
    ]
    # Placeholder zeros.
    for col in expected_cols[1:]:
        assert df[col].tolist() == [0, 0]


def test_centering_prep_2d_input_supported(tmp_path, minimal_sphere_header):
    """A 2D image input is accepted (legacy supports both 2D and 3D)."""
    config = _make_config(tmp_path, overwrite=True, make_plots=False)
    output_dir = tmp_path / "out"

    image = np.zeros((64, 64), dtype=np.float32)
    f_path = tmp_path / "single_frame.fits"
    fits.writeto(f_path, image, header=minimal_sphere_header, overwrite=True)

    CenteringPreparer(
        config=config,
        input_files=[f_path],
        output_dir=output_dir,
    ).run()

    csv_path = output_dir / "centers.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert df["frames"].tolist() == ["single_frame.fits"]


def test_centering_prep_4d_input_raises(tmp_path, minimal_sphere_header):
    """Unsupported dimensionality (4D) -> ShapeMismatchError."""
    config = _make_config(tmp_path, overwrite=True, make_plots=False)
    output_dir = tmp_path / "out"

    cube4 = np.zeros((2, 2, 8, 8), dtype=np.float32)
    f_path = tmp_path / "weird.fits"
    fits.writeto(f_path, cube4, header=minimal_sphere_header, overwrite=True)

    step = CenteringPreparer(
        config=config,
        input_files=[f_path],
        output_dir=output_dir,
    )

    with pytest.raises(ShapeMismatchError):
        step.run()


def test_centering_prep_missing_input(tmp_path):
    """Missing input file -> StepExecutionError."""
    config = _make_config(tmp_path, overwrite=True, make_plots=False)
    output_dir = tmp_path / "out"

    step = CenteringPreparer(
        config=config,
        input_files=[tmp_path / "ghost.fits"],
        output_dir=output_dir,
    )
    with pytest.raises(StepExecutionError):
        step.run()


def test_centering_prep_empty_input(tmp_path):
    """Empty input list -> InvalidConfigError."""
    config = _make_config(tmp_path, overwrite=True, make_plots=False)
    output_dir = tmp_path / "out"
    with pytest.raises(InvalidConfigError):
        CenteringPreparer(
            config=config,
            input_files=[],
            output_dir=output_dir,
        ).run()


def test_centering_prep_resumability_augments_csv(
    tmp_path, minimal_sphere_header
):
    """Existing centers.csv + overwrite=False: existing rows preserved.

    User-edited values for frame_001 must NOT be reset; frame_002 is added
    with placeholder zeros.
    """
    config = _make_config(tmp_path, overwrite=False, make_plots=False)
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True)

    # Pre-populate CSV with hand-edited values for frame_001.
    pre_df = pd.DataFrame(
        {
            "frames": ["frame_001.fits"],
            "x1": [100], "y1": [200],
            "x2": [101], "y2": [201],
            "x3": [102], "y3": [202],
            "x4": [103], "y4": [203],
        }
    )
    csv_path = output_dir / "centers.csv"
    pre_df.to_csv(csv_path, index=False)

    f1 = _write_cube(tmp_path, "frame_001.fits", 2, minimal_sphere_header)
    f2 = _write_cube(tmp_path, "frame_002.fits", 2, minimal_sphere_header)

    CenteringPreparer(
        config=config,
        input_files=[f1, f2],
        output_dir=output_dir,
    ).run()

    df = pd.read_csv(csv_path)
    # Original hand edits preserved.
    row1 = df[df["frames"] == "frame_001.fits"].iloc[0]
    assert int(row1["x1"]) == 100
    assert int(row1["y4"]) == 203
    # New row with zero placeholders.
    row2 = df[df["frames"] == "frame_002.fits"].iloc[0]
    for col in ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]:
        assert int(row2[col]) == 0


def test_centering_prep_no_duplicate_rows_on_repeat(
    tmp_path, minimal_sphere_header
):
    """Re-running with the same files + overwrite=False does not duplicate rows.

    Regression guard for the legacy ``not file in DATAFRAME['frames']`` bug
    (Series ``__contains__`` checks the index, not values).
    """
    config = _make_config(tmp_path, overwrite=False, make_plots=False)
    output_dir = tmp_path / "out"

    f1 = _write_cube(tmp_path, "frame_001.fits", 2, minimal_sphere_header)

    step = CenteringPreparer(
        config=config,
        input_files=[f1],
        output_dir=output_dir,
    )
    step.run()
    step.run()  # second run should not append a duplicate row.

    df = pd.read_csv(output_dir / "centers.csv")
    assert len(df) == 1
    assert df["frames"].tolist() == ["frame_001.fits"]
