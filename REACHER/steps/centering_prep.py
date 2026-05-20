"""Centering preparation step.

This refactors :func:`REACHER.routines.prepare_centering`. Per file, it
generates a quick-look PDF that overlays the detector center, a 1D
Gaussian centroid, and the four nominal satellite-spot positions to help
the operator hand-edit the resulting ``centers.csv`` file (the manual
human-in-the-loop step described in Architecture Rule 12).

After processing all input files the step ensures a ``centers.csv`` file
exists in ``output_dir`` containing one row per input frame with placeholder
satellite-spot coordinates (zeros) that the user is expected to fill in.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from astropy.io import fits

from ..config import ReacherConfig
from ..exceptions import (
    InvalidConfigError,
    ShapeMismatchError,
    StepExecutionError,
)
from .base import PipelineStep


# Column names of the centers.csv file. Frame name + four (x, y) sat spots.
_CENTERS_COLUMNS = [
    "frames",
    "x1",
    "y1",
    "x2",
    "y2",
    "x3",
    "y3",
    "x4",
    "y4",
]


class CenteringPreparer(PipelineStep):
    """Prepare centering helper PDFs and the ``centers.csv`` placeholder file.

    The algorithm preserves the legacy
    :func:`routines.prepare_centering` behavior:

    1. For each input file, generate a quick-look PDF (when
       ``config.processing.make_plots`` is true) showing the detector
       center, a 1D Gaussian centroid, and the four nominal satellite
       spot positions.
    2. Build / update ``output_dir/centers.csv`` so that every input file
       has a row with zero-initialized satellite-spot columns. Existing
       rows are preserved (this is the human-in-the-loop hand-off file).

    Parameters
    ----------
    config
        Reacher configuration object. ``processing.make_plots`` controls
        whether the per-file PDFs are generated. ``processing.overwrite``
        controls resumability of ``centers.csv``.
    input_files
        Sequence of FITS files to inspect.
    output_dir
        Directory where ``centers.csv`` and ``recentering_pngs/`` are
        written.

    Raises
    ------
    InvalidConfigError
        If ``input_files`` is empty.
    StepExecutionError
        If any input file is missing or contains no data.
    ShapeMismatchError
        If an input file has unsupported dimensionality (not 2D / 3D).

    Notes
    -----
    Behavior change vs legacy (per Architecture Rule 7):

    * The legacy routine used ``if not file in DATAFRAME['frames']`` to
      detect missing rows when augmenting an existing CSV. Indexing a
      pandas Series with ``in`` checks the **index** rather than the
      values, which silently caused duplicate rows to be appended on
      every run. We use ``DATAFRAME['frames'].values`` instead, which
      matches the legacy *intent*.
    * Plot generation is gated on ``config.processing.make_plots``. The
      legacy routine always plotted; making it conditional is consistent
      with the rest of the pipeline and makes the step testable
      headlessly.
    * The skip behavior honors ``processing.overwrite``: when
      ``centers.csv`` exists and ``overwrite=False`` it is augmented in
      place (legacy behavior), but when ``overwrite=True`` the CSV is
      regenerated from scratch.
    """

    def __init__(
        self,
        config: ReacherConfig,
        input_files: Sequence[str | Path],
        output_dir: str | Path,
    ) -> None:
        super().__init__(config)
        self.input_files = [Path(p) for p in input_files]
        self.output_dir = Path(output_dir)

    def _check_inputs(self) -> None:
        if not self.input_files:
            raise InvalidConfigError(
                "CenteringPreparer requires at least one input file"
            )
        missing = [str(p) for p in self.input_files if not p.exists()]
        if missing:
            raise StepExecutionError(
                f"Missing input files: {', '.join(missing)}"
            )

    def run(self) -> None:
        """Execute the centering preparation step."""
        self._check_inputs()
        if not self._quiet:
            self.logger.info("Starting CenteringPreparer")
        self.logger.debug(
            "Input files: %s",
            ", ".join(p.name for p in self.input_files),
        )
        self.logger.debug("Output dir: %s", self.output_dir.name)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.processing.make_plots:
            self._make_plots()
        else:
            # Still validate that each file is readable and has acceptable
            # dimensionality, so the centers.csv stage doesn't drift.
            for file_path in self.input_files:
                self._validate_file(file_path)

        self._write_centers_csv()

        if not self._quiet:
            self.logger.info("Finished CenteringPreparer")

    # ------------------------------------------------------------------
    # Per-file processing
    # ------------------------------------------------------------------
    def _validate_file(self, file_path: Path) -> np.ndarray:
        """Load + dimensionality-check a single file.

        Mirrors the legacy frame-selection rule:

        * 3D cube with ``len > 1`` -> use frame index 1.
        * 3D cube with ``len == 1`` -> use frame index 0.
        * 2D image -> use directly.
        * Anything else -> :class:`ShapeMismatchError`.
        """
        with fits.open(file_path) as hdul:
            data = hdul[0].data

        if data is None:
            raise StepExecutionError(
                f"Input file {file_path.name} contains no data"
            )

        if data.ndim == 3:
            if data.shape[0] > 1:
                image = data[1]
            else:
                image = data[0]
        elif data.ndim == 2:
            image = data
        else:
            raise ShapeMismatchError(
                "CenteringPreparer expects 2D or 3D data; "
                f"got ndim={data.ndim} (shape={data.shape}) for "
                f"{file_path.name}"
            )

        return np.asarray(image)

    def _make_plots(self) -> None:
        """Render the per-file centering quick-look PDFs.

        Imports of matplotlib / scipy / photutils / VIP are deferred so
        that the rest of the pipeline (and the unit tests with
        ``make_plots=False``) can run without those heavy dependencies
        being healthy.
        """
        import matplotlib

        # Force a non-interactive backend for headless / test environments.
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from photutils.centroids import centroid_1dg
        from scipy import ndimage
        from vip_hci.var import frame_center

        from ..plots import show_img

        png_dir = self.output_dir / "recentering_pngs"
        png_dir.mkdir(parents=True, exist_ok=True)

        for file_path in self.input_files:
            image = self._validate_file(file_path)

            data_blurred = ndimage.gaussian_filter(image.astype(float), sigma=10)
            try:
                center_blurred = centroid_1dg(data_blurred)
            except Exception as exc:  # pragma: no cover - plotting path
                self.logger.warning(
                    "centroid_1dg failed for %s: %s; falling back to "
                    "frame center",
                    file_path.name,
                    exc,
                )
                cy_fallback, cx_fallback = frame_center(image)
                center_blurred = (cx_fallback, cy_fallback)

            cy, cx = frame_center(image)
            sat_spots = (
                np.array([[-35, 35], [35, 35], [-35, -35], [35, -35]])
                + np.asarray(center_blurred)
            )

            show_img(
                image,
                minper=0.1,
                maxper=99.9,
                title=file_path.name,
                xlim=(cx - 70, cx + 70),
                ylim=(cy - 70, cy + 70),
                size=5,
                dpi=120,
            )

            plt.scatter(
                center_blurred[0], center_blurred[1],
                color="black", s=200, marker="+",
            )
            plt.scatter(cx, cy, color="black", s=200, marker="*")
            plt.scatter(
                sat_spots[:, 0], sat_spots[:, 1],
                marker="o", color="black", s=400, lw=2, facecolor="none",
            )
            for i in range(len(sat_spots)):
                plt.text(
                    sat_spots[i, 0] + 7, sat_spots[i, 1] + 7,
                    f"{i + 1}", color="black", fontsize=16,
                )
            plt.scatter([], [], color="black", label="center frame", marker="*")
            plt.scatter([], [], color="black", label="centr 1D Gauss", marker="+")
            plt.scatter(
                [], [], marker="o", color="black",
                label="sat-spots", facecolor="none",
            )
            plt.legend()

            pdf_path = png_dir / f"{file_path.stem}.pdf"
            plt.savefig(
                pdf_path, dpi=120, format="pdf", bbox_inches="tight"
            )
            plt.close()

    # ------------------------------------------------------------------
    # CSV management
    # ------------------------------------------------------------------
    def _write_centers_csv(self) -> None:
        """Create or augment the ``centers.csv`` placeholder file."""
        csv_path = self.output_dir / "centers.csv"
        frame_names = [f.name for f in self.input_files]

        if csv_path.exists() and not self.config.processing.overwrite:
            self.logger.info(
                "%s exists; augmenting in place", csv_path.name
            )
            df = pd.read_csv(csv_path)
            existing = set(df["frames"].values)
            new_rows = [name for name in frame_names if name not in existing]
            if new_rows:
                addition = pd.DataFrame(
                    {
                        "frames": new_rows,
                        **{
                            col: [0] * len(new_rows)
                            for col in _CENTERS_COLUMNS
                            if col != "frames"
                        },
                    }
                )
                df = pd.concat([df, addition], ignore_index=True)
        else:
            df = pd.DataFrame(
                {
                    "frames": frame_names,
                    **{
                        col: [0] * len(frame_names)
                        for col in _CENTERS_COLUMNS
                        if col != "frames"
                    },
                }
            )

        df = df[_CENTERS_COLUMNS]
        df.to_csv(csv_path, index=False)
