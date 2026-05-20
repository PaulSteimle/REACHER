"""Channel splitting step: split an IRDIS science cube into LEFT/RIGHT.

This refactors :func:`REACHER.routines.split_file_subtract_sky`. Despite the
legacy function name, that function does **not** perform any sky
subtraction; it simply splits the IRDIS detector into its LEFT and RIGHT
channels using fixed pixel ranges. Sky subtraction is the responsibility of
the upcoming :class:`MasterSkyCreator` / :class:`CubeAssembler` steps. We
therefore name this step ``ChannelSplitter``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from ..config import ReacherConfig
from ..exceptions import (
    InvalidConfigError,
    ShapeMismatchError,
    StepExecutionError,
)
from .base import PipelineStep


class ChannelSplitter(PipelineStep):
    """Split an IRDIS science cube into its LEFT and RIGHT detector channels.

    The algorithm preserves :func:`routines.split_file_subtract_sky`:

    1. If both ``left_output_file`` and ``right_output_file`` already exist
       and ``overwrite=False`` the step is skipped.
    2. The science cube is loaded.
    3. The LEFT and RIGHT channels are sliced using the per-channel
       ``rows`` / ``cols`` slices on
       ``config.instrument.detector`` (the SPHERE preset matches the
       module-level :data:`IRDIS_LEFT_CHANNEL` /
       :data:`IRDIS_RIGHT_CHANNEL`). The LEFT and RIGHT pixel ranges
       differ by design — this is detector geometry, not a bug
       (Architecture Rule 11).
    4. Each channel is written to its own FITS file with the header of the
       input file passed through unchanged (matches the legacy behavior).

    Parameters
    ----------
    config
        Reacher configuration object.
    input_file
        Path to the input science FITS cube. Must be a 3D cube — the
        legacy slice ``data[:, ...]`` requires ``ndim == 3``.
    left_output_file
        Path where the LEFT channel FITS file will be written.
    right_output_file
        Path where the RIGHT channel FITS file will be written.

    Raises
    ------
    InvalidConfigError
        If a required path argument is missing.
    StepExecutionError
        If the input file is missing or contains no data.
    ShapeMismatchError
        If the input data is not 3D.

    Notes
    -----
    The input header is preserved as-is (no sanitization), matching the
    legacy routine. ``astropy`` will emit ``VerifyWarning`` for HIERARCH
    cards on write — benign, and consistent with the other steps.
    """

    def __init__(
        self,
        config: ReacherConfig,
        input_file: str | Path,
        left_output_file: str | Path,
        right_output_file: str | Path,
    ) -> None:
        super().__init__(config)
        self.input_file = Path(input_file)
        self.left_output_file = Path(left_output_file)
        self.right_output_file = Path(right_output_file)

    def _check_inputs(self) -> None:
        if self.input_file is None:
            raise InvalidConfigError("ChannelSplitter requires input_file")
        if self.left_output_file is None:
            raise InvalidConfigError(
                "ChannelSplitter requires left_output_file"
            )
        if self.right_output_file is None:
            raise InvalidConfigError(
                "ChannelSplitter requires right_output_file"
            )

        if not self.input_file.exists():
            raise StepExecutionError(
                f"Missing input science file: {self.input_file}"
            )

    def _both_outputs_exist(self) -> bool:
        """Skip helper: both LEFT and RIGHT outputs must be present."""
        if self.config.processing.overwrite:
            return False
        if self.left_output_file.exists() and self.right_output_file.exists():
            self.logger.info(
                "%s and %s exist; skipping",
                self.left_output_file.name,
                self.right_output_file.name,
            )
            return True
        return False

    def run(self) -> None:
        """Execute the channel split step."""
        self._check_inputs()
        if not self._quiet:
            self.logger.info("Starting ChannelSplitter")
        self.logger.debug(
            "Input: %s -> LEFT: %s, RIGHT: %s",
            self.input_file.name,
            self.left_output_file.name,
            self.right_output_file.name,
        )

        self.left_output_file.parent.mkdir(parents=True, exist_ok=True)
        self.right_output_file.parent.mkdir(parents=True, exist_ok=True)

        if self._both_outputs_exist():
            return

        with fits.open(self.input_file) as hdul:
            data = hdul[0].data
            header = hdul[0].header.copy()

        if data is None:
            raise StepExecutionError(
                f"Science file {self.input_file.name} contains no data"
            )

        if data.ndim != 3:
            raise ShapeMismatchError(
                "ChannelSplitter expects a 3D cube; "
                f"got ndim={data.ndim} (shape={data.shape}) for "
                f"{self.input_file.name}"
            )

        # LEFT and RIGHT channel slices are instrument-coupled and pulled
        # from the detector profile on ``config.instrument.detector``. The
        # LEFT and RIGHT row slices differ by detector geometry (the right
        # channel is offset by ~10 px in y) -- this is not a bug.
        left_channel = self.config.instrument.detector.left_channel
        right_channel = self.config.instrument.detector.right_channel
        rows_l, cols_l = left_channel["rows"], left_channel["cols"]
        rows_r, cols_r = right_channel["rows"], right_channel["cols"]

        left_frame = np.asarray(data)[:, rows_l, cols_l]
        right_frame = np.asarray(data)[:, rows_r, cols_r]

        self._write_channel(self.left_output_file, left_frame, header)
        self._write_channel(self.right_output_file, right_frame, header)

        if not self._quiet:
            self.logger.info("Finished ChannelSplitter")

    def _write_channel(
        self, path: Path, data: np.ndarray, header: fits.Header
    ) -> None:
        """Write a channel FITS file passing the input header through unchanged.

        The legacy routine assigned ``hdul[0].header = hdul_input[0].header``
        directly, so we mirror that without sanitization.
        """
        hdu = fits.PrimaryHDU(data=data)
        hdu.header = header
        fits.HDUList([hdu]).writeto(path, overwrite=True)
