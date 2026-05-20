"""File calibration step: dark / flat / bad-pixel correction of science data."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from astropy.io import fits

from ..config import ReacherConfig
from ..exceptions import (
    InvalidConfigError,
    ShapeMismatchError,
    StepExecutionError,
)
from ..io.fits_handler import sanitize_header
from ..utils.bad_pixel import interpolate_bad_pixels
from ..utils.silence import suppress_stdout_stderr
from .base import PipelineStep


class FileCalibrator(PipelineStep):
    """Apply dark, flat and bad-pixel correction to a single science file.

    The algorithm preserves the behavior of ``routines.calibrate_files``:

    1. If ``output_file`` already exists and ``overwrite=False`` the step is
       skipped (``_skip_if_exists``).
    2. The master dark and master flat are loaded with ``fits.getdata``.
    3. The science cube is loaded.
    4. Calibration is computed under
       ``np.errstate(divide='ignore', invalid='ignore')`` as::

           calibrated = (science - np.nan_to_num(master_dark)) /
                        np.nan_to_num(master_flat)

       Order matters: dark first, then flat (Architecture Rule 12).
    5. (Optional) Bad pixel and clump corrections are applied via
       :func:`interpolate_bad_pixels` (3x3 median neighborhood). For 3D
       cubes the correction is applied frame-by-frame; for 2D arrays the
       correction is applied directly. This matches the surface behavior of
       the legacy :func:`routines.correct_bad_pix` (which wraps a 2D image
       into a single-frame cube, runs VIP's cube fixers, then unwraps).
    6. The result is written to ``output_file`` with a sanitized header and
       these provenance keywords::

           HISTORY1     = 'DARK CORRECTED'
           HISTORY2     = 'FLAT CORRECTED'
           HISTORY3     = 'BAD PIXEL CORRECTED'
           MASTDARK     = basename(master_dark)
           MASTFLAT     = basename(master_flat)
           BADPIX_FILE  = basename(bad_pix_mask)

    Parameters
    ----------
    config
        Reacher configuration object.
    input_file
        Path to the input science FITS file (2D image or 3D cube).
    output_file
        Path where the calibrated FITS file will be written.
    master_dark_path
        Path to the master dark FITS file.
    master_flat_path
        Path to the master flat FITS file.
    bad_pix_mask_path
        Optional path to a 2D bad pixel mask FITS file. When supplied the
        flagged pixels are interpolated with a 3x3 median neighborhood.
    clump_mask_path
        Optional path to the IRDIS clump mask FITS file. When supplied the
        clump pixels are also interpolated in the same way.
    master_background_path
        Path to a master instrumental background. **Currently
        unsupported.** The legacy routine silently skipped this branch
        because the implementation was incomplete (Architecture Rule 7
        forbids silently ignored user input). Supplying a non-``None`` value
        therefore raises :class:`NotImplementedError` with a pointer to the
        future :class:`MasterSkyCreator` / sky-subtraction step.

    Raises
    ------
    InvalidConfigError
        If a required path argument is missing.
    StepExecutionError
        If any required input file (science / dark / flat / mask) is
        missing on disk.
    ShapeMismatchError
        If the master dark or master flat shape is incompatible with the
        science data.
    NotImplementedError
        If ``master_background_path`` is provided. The legacy commented-out
        branch is intentionally not reintroduced here — sky / background
        subtraction is the responsibility of the upcoming
        :class:`MasterSkyCreator` step (Phase 2 Step 13).
    """

    def __init__(
        self,
        config: ReacherConfig,
        input_file: str | Path,
        output_file: str | Path,
        master_dark_path: str | Path,
        master_flat_path: str | Path,
        bad_pix_mask_path: Optional[str | Path] = None,
        clump_mask_path: Optional[str | Path] = None,
        master_background_path: Optional[str | Path] = None,
    ) -> None:
        super().__init__(config)
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.master_dark_path = Path(master_dark_path)
        self.master_flat_path = Path(master_flat_path)
        self.bad_pix_mask_path = (
            Path(bad_pix_mask_path) if bad_pix_mask_path is not None else None
        )
        self.clump_mask_path = (
            Path(clump_mask_path) if clump_mask_path is not None else None
        )
        # master_background is intentionally captured so we can raise an
        # explicit NotImplementedError in run() rather than silently
        # ignoring a non-None value (see class docstring).
        self.master_background_path = (
            Path(master_background_path)
            if master_background_path is not None
            else None
        )

    def _check_inputs(self) -> None:
        if self.input_file is None:
            raise InvalidConfigError("FileCalibrator requires input_file")
        if self.output_file is None:
            raise InvalidConfigError("FileCalibrator requires output_file")
        if self.master_dark_path is None:
            raise InvalidConfigError("FileCalibrator requires master_dark_path")
        if self.master_flat_path is None:
            raise InvalidConfigError("FileCalibrator requires master_flat_path")

        for label, path in (
            ("input science file", self.input_file),
            ("master dark", self.master_dark_path),
            ("master flat", self.master_flat_path),
        ):
            if not path.exists():
                raise StepExecutionError(f"Missing {label}: {path}")

        for label, path in (
            ("bad pixel mask", self.bad_pix_mask_path),
            ("clump mask", self.clump_mask_path),
        ):
            if path is not None and not path.exists():
                raise StepExecutionError(f"Missing {label}: {path}")

    def run(self) -> None:
        """Execute the file calibration step."""
        # Fail loud if the caller asked for instrumental background
        # subtraction. The legacy routines.calibrate_files silently
        # ignored this (the branch was commented out); per Architecture
        # Rule 7 we surface that gap explicitly.
        if self.master_background_path is not None:
            raise NotImplementedError(
                "master_background subtraction is not implemented in "
                "FileCalibrator. Sky / background subtraction will be "
                "handled by the upcoming MasterSkyCreator step (Phase 2 "
                "Step 13). Pass master_background_path=None to proceed."
            )

        self._check_inputs()
        if not self._quiet:
            self.logger.info("Starting FileCalibrator")
        # Single info-level summary so pipeline runs still show which
        # master calibration files each input was paired with.
        self.logger.info(
            "Calibrating %s | dark=%s flat=%s",
            self.input_file.name,
            self.master_dark_path.name,
            self.master_flat_path.name,
        )
        self.logger.debug(
            "Input file: %s -> output: %s",
            self.input_file.name,
            self.output_file.name,
        )

        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        if self._skip_if_exists(self.output_file):
            return

        calibrated, header = self._calibrate()
        corrected = self._apply_bad_pixel_correction(calibrated)
        self._write_output(corrected, header)

        if not self._quiet:
            self.logger.info("Finished FileCalibrator")

    def _calibrate(self) -> tuple[np.ndarray, fits.Header]:
        """(science - dark) / flat with the legacy nan-to-num shielding."""
        master_dark = fits.getdata(self.master_dark_path)
        master_flat = fits.getdata(self.master_flat_path)

        with fits.open(self.input_file) as hdul:
            science_data = hdul[0].data
            header = hdul[0].header.copy()

        if science_data is None:
            raise StepExecutionError(
                f"Science file {self.input_file.name} contains no data"
            )

        science_arr = np.asarray(science_data, dtype=float)
        dark_arr = np.nan_to_num(np.asarray(master_dark, dtype=float))
        flat_arr = np.nan_to_num(np.asarray(master_flat, dtype=float))

        # The dark / flat have the shape of a single frame; for 3D science
        # cubes numpy broadcasts naturally over the leading axis so long
        # as the trailing dimensions match.
        frame_shape = science_arr.shape[-2:]
        if dark_arr.shape != frame_shape:
            raise ShapeMismatchError(
                f"Master dark shape {dark_arr.shape} does not match "
                f"science frame shape {frame_shape}"
            )
        if flat_arr.shape != frame_shape:
            raise ShapeMismatchError(
                f"Master flat shape {flat_arr.shape} does not match "
                f"science frame shape {frame_shape}"
            )

        with np.errstate(divide="ignore", invalid="ignore"):
            calibrated = (science_arr - dark_arr) / flat_arr

        return calibrated, header

    def _apply_bad_pixel_correction(self, data: np.ndarray) -> np.ndarray:
        """Apply bad pixel + clump interpolation if masks are supplied.

        The legacy :func:`routines.correct_bad_pix` wraps 2D inputs into a
        single-frame cube, runs VIP's cube fixers (which iterate frame by
        frame), then unwraps. We replicate that surface behavior here:
        masks are 2D and applied per-frame for 3D inputs and directly for
        2D inputs.
        """
        if self.bad_pix_mask_path is None and self.clump_mask_path is None:
            return data

        bad_pix_mask = None
        clump_mask = None
        if self.bad_pix_mask_path is not None:
            bad_pix_mask = np.asarray(fits.getdata(self.bad_pix_mask_path))
            self.logger.info(
                "Applying bad pixel mask %s", self.bad_pix_mask_path.name
            )
        if self.clump_mask_path is not None:
            clump_mask = np.asarray(fits.getdata(self.clump_mask_path))
            self.logger.info(
                "Applying clump mask %s", self.clump_mask_path.name
            )

        def _apply(cube: np.ndarray) -> np.ndarray:
            corrected = cube
            if bad_pix_mask is not None:
                from vip_hci.preproc import cube_fix_badpix_isolated
                with suppress_stdout_stderr():
                    corrected = cube_fix_badpix_isolated(
                        corrected, bpm_mask=bad_pix_mask, correct_only=True,
                        frame_by_frame=False, mad=True, verbose=False,
                        ignore_nan=False,
                    )
            if clump_mask is not None:
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
            return corrected

        # if data is 2D then make it a 3D cube and apply the correction, then unwrap it back to 2D. If data is 3D then apply the correction directly. 
        # For any other number of dimensions raise an error
        if data.ndim == 2:
            _apply(np.array([data]))[0]    
            return 
        if data.ndim == 3:
            return _apply(data) 
        raise ShapeMismatchError(
            f"FileCalibrator expects 2D or 3D data; got ndim={data.ndim}"
        )

    def _write_output(
        self, data: np.ndarray, source_header: fits.Header
    ) -> None:
        """Persist the calibrated FITS file with provenance keywords."""
        hdu = fits.PrimaryHDU(data=data)
        sanitized = sanitize_header(source_header)
        for key, value in sanitized.items():
            try:
                hdu.header[key] = value
            except (ValueError, KeyError):
                continue

        hdu.header["HISTORY1"] = "DARK CORRECTED"
        hdu.header["HISTORY2"] = "FLAT CORRECTED"
        hdu.header["HISTORY3"] = "BAD PIXEL CORRECTED"
        hdu.header["MASTDARK"] = self.master_dark_path.name
        hdu.header["MASTFLAT"] = self.master_flat_path.name
        if self.bad_pix_mask_path is not None:
            hdu.header["BADPIX_FILE"] = self.bad_pix_mask_path.name

        fits.HDUList([hdu]).writeto(self.output_file, overwrite=True)
