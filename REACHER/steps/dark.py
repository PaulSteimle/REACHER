"""Master dark creation step."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from astropy.io import fits

from ..config import ReacherConfig
from ..exceptions import StepExecutionError, InvalidConfigError
from ..io.fits_handler import sanitize_header
from .base import PipelineStep
from ..utils.bad_pixel import interpolate_bad_pixels
from ..utils.formatting import format_duration
from ..utils.silence import suppress_stdout_stderr


class MasterDarkCreator(PipelineStep):
    """Build a master dark by median-combining input dark files.

    The algorithm preserves the behavior of ``routines.make_master_dark``:

    1. For each input dark file the FITS data is loaded.
       - 2D arrays are appended as-is.
       - 3D cubes are first collapsed via ``np.nanmedian`` along axis 0.
    2. The per-file frames are stacked and median combined along axis 0.
    3. (Optional) Bad pixels and clump regions are interpolated when mask
       arrays/paths are supplied.
    4. The result is written to ``master_dark_path`` with a sanitized header,
       a ``HISTORY`` of ``MASTER DARK`` and ``DARK0..DARKN`` keywords listing
       the input file basenames.

    Parameters
    ----------
    config
        Reacher configuration object.
    dark_files
        Sequence of input dark FITS file paths.
    master_dark_path
        Output FITS path for the master dark.
    bad_pix_mask_path
        Optional path to a 2D bad pixel mask FITS file.  When provided, bad
        pixels in the median-combined dark are interpolated using a 3x3
        median neighborhood (see :func:`interpolate_bad_pixels`).
    clump_mask_path
        Optional path to the IRDIS clump mask FITS file.  When provided the
        clump pixels are also interpolated in the same way.

    Raises
    ------
    InvalidConfigError
        If ``dark_files`` is empty.
    StepExecutionError
        If any of the input files (darks or masks) cannot be located, or if
        no usable frames could be read from the supplied dark files.
    """

    def __init__(
        self,
        config: ReacherConfig,
        dark_files: Sequence[str | Path],
        master_dark_path: str | Path,
        bad_pix_mask_path: Optional[str | Path] = None,
        clump_mask_path: Optional[str | Path] = None,
    ) -> None:
        super().__init__(config)
        self.dark_files = [Path(p) for p in dark_files]
        self.master_dark_path = Path(master_dark_path)
        self.bad_pix_mask_path = (
            Path(bad_pix_mask_path) if bad_pix_mask_path is not None else None
        )
        self.clump_mask_path = (
            Path(clump_mask_path) if clump_mask_path is not None else None
        )

    def _check_inputs(self) -> None:
        if not self.dark_files:
            raise InvalidConfigError("No dark files provided for master dark")

        missing = [str(p) for p in self.dark_files if not p.exists()]
        if missing:
            raise StepExecutionError(
                f"Missing dark files: {', '.join(missing)}"
            )

        for label, path in (
            ("bad pixel mask", self.bad_pix_mask_path),
            ("clump mask", self.clump_mask_path),
        ):
            if path is not None and not path.exists():
                raise StepExecutionError(f"Missing {label}: {path}")

    def run(self) -> None:
        """Execute the master dark creation step."""
        t0 = time.perf_counter()
        self._check_inputs()
        if not self._quiet:
            self.logger.info("Starting MasterDarkCreator")
        self.logger.debug(
            "Dark files: %s", ", ".join(p.name for p in self.dark_files)
        )
        self.logger.debug("Master dark output: %s", self.master_dark_path.name)

        self.master_dark_path.parent.mkdir(parents=True, exist_ok=True)

        if self._skip_if_exists(self.master_dark_path):
            return

        median_dark, header = self._build_median_dark()
        master_dark = self._apply_bad_pixel_correction(median_dark)
        self._write_master_dark(master_dark, header)

        self.logger.info(
            "Created %s in %s",
            self.master_dark_path.name,
            format_duration(time.perf_counter() - t0),
        )
        if not self._quiet:
            self.logger.info("Finished MasterDarkCreator")

    def _build_median_dark(self) -> tuple[np.ndarray, fits.Header]:
        """Read each dark file and median combine them frame by frame."""
        per_file_frames: list[np.ndarray] = []
        last_header: Optional[fits.Header] = None

        for dark_path in self.dark_files:
            with fits.open(dark_path) as hdul:
                data = hdul[0].data
                last_header = hdul[0].header.copy()

            if data is None:
                self.logger.warning(
                    "%s has no data; skipping", dark_path.name
                )
                continue

            if data.ndim == 2:
                per_file_frames.append(np.asarray(data, dtype=float))
            elif data.ndim == 3:
                exptime = (
                    last_header.get("EXPTIME") if last_header is not None else None
                )
                self.logger.info(
                    "Dark %ss exists of multiple exposures, "
                    "combining to median file",
                    exptime,
                )
                per_file_frames.append(
                    np.nanmedian(np.asarray(data, dtype=float), axis=0)
                )
            else:
                self.logger.warning(
                    "%s has unexpected ndim=%d; skipping",
                    dark_path.name,
                    data.ndim,
                )

        if not per_file_frames:
            raise StepExecutionError(
                "No usable dark frames found for master dark"
            )

        stacked = np.array(per_file_frames)
        median_dark = np.nanmedian(stacked, axis=0)

        if last_header is None:
            last_header = fits.Header()

        return median_dark, last_header

    def _apply_bad_pixel_correction(self, median_dark: np.ndarray) -> np.ndarray:
        """Interpolate bad pixel and clump locations if masks are supplied."""
        corrected = np.array([median_dark])


        if self.bad_pix_mask_path is not None:
            bad_pix_mask = fits.getdata(self.bad_pix_mask_path)
            self.logger.info(
                "Applying bad pixel mask %s", self.bad_pix_mask_path.name
            )
            from vip_hci.preproc import cube_fix_badpix_isolated
            with suppress_stdout_stderr():
                corrected = cube_fix_badpix_isolated(
                    corrected, bpm_mask=bad_pix_mask, correct_only=True,
                    frame_by_frame=False, mad=True, verbose=False,
                    ignore_nan=False,
                )

         # then the clumps
        if self.clump_mask_path is not None:
            clump_mask = fits.getdata(self.clump_mask_path)
            self.logger.info(
                "Applying clump mask %s", self.clump_mask_path.name
            )
            from vip_hci.preproc import cube_fix_badpix_interp
            # first set the areas to nans to make sure they do not contaminate
            nan_mask = np.ones_like(clump_mask)
            nan_mask[clump_mask == 1] = np.nan
            corrected *= nan_mask
            with suppress_stdout_stderr():
                corrected = cube_fix_badpix_interp(
                    corrected, bpm_mask=clump_mask, mode='gauss',
                    kernel_sz=35,
                )

        return corrected[0]


    def _write_master_dark(
        self, master_dark: np.ndarray, source_header: fits.Header
    ) -> None:
        """Persist the master dark FITS file with provenance keywords."""
        hdu = fits.PrimaryHDU(data=master_dark)
        # sanitize_header returns a dict of cleaned keys (spaces -> underscores)
        sanitized = sanitize_header(source_header)
        for key, value in sanitized.items():
            try:
                hdu.header[key] = value
            except (ValueError, KeyError):
                # Skip keywords that astropy refuses (too long, reserved, etc.)
                continue

        hdu.header["HISTORY"] = "MASTER DARK"
        for i, dark_path in enumerate(self.dark_files):
            hdu.header[f"DARK{i}"] = dark_path.name

        fits.HDUList([hdu]).writeto(self.master_dark_path, overwrite=True)
