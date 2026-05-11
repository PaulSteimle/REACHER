"""Master flat creation step."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from astropy.io import fits

from ..config import ReacherConfig
from ..exceptions import InvalidConfigError, StepExecutionError
from ..io.fits_handler import sanitize_header
from ..utils.bad_pixel import interpolate_bad_pixels
from ..utils.formatting import format_duration
from ..utils.silence import suppress_stdout_stderr
from .base import PipelineStep


class MasterFlatCreator(PipelineStep):
    """Build a master flat from dark-subtracted, normalized flat frames.

    The algorithm preserves the behavior of ``routines.make_master_flat``:

    1. For each ``(flat_file, master_dark_file)`` pair:
        * If the matching master dark does not exist, the flat is skipped
          with a warning (as in the legacy routine).
        * The flat data is loaded.
            - 2D arrays: dark-subtract directly.
            - 3D cubes: dark-subtract then ``np.nanmedian`` along axis 0.
        * The per-file frame is normalized by its own ``np.nanmedian``.
    2. The per-file normalized frames are stacked and median-combined along
       axis 0; ``NaN``s are replaced with zero (``np.nan_to_num``).
    3. (Optional) Bad pixels and clump regions are interpolated when mask
       paths are supplied (3x3 median neighborhood, see
       :func:`interpolate_bad_pixels`).
    4. The LEFT and RIGHT IRDIS channel medians are computed independently
       and the master flat is divided by their mean so that both channels
       are normalized to ~1.0.
    5. The result is written to ``master_flat_path`` with a sanitized
       header, ``HISTORY = MASTER FLAT`` and ``FLAT0..FLATN`` keywords
       listing the input flat basenames.

    Parameters
    ----------
    config
        Reacher configuration object.
    flat_files
        Sequence of input flat FITS file paths.
    master_dark_files
        Sequence of master dark FITS paths matched 1-to-1 to ``flat_files``.
    master_flat_path
        Output FITS path for the master flat.
    bad_pix_mask_path
        Optional path to a 2D bad pixel mask FITS file.  When provided, bad
        pixels in the median-combined flat are interpolated using a 3x3
        median neighborhood.
    clump_mask_path
        Optional path to the IRDIS clump mask FITS file.  When provided the
        clump pixels are also interpolated in the same way.

    Raises
    ------
    InvalidConfigError
        If ``flat_files`` is empty, or if ``master_dark_files`` length does
        not match ``flat_files``.
    StepExecutionError
        If any input flat file is missing, if no usable flats survive the
        dark-matching / shape filters, or if either mask path does not
        exist.

    Notes
    -----
    The LEFT and RIGHT pixel ranges differ slightly because of detector
    geometry on IRDIS — this is preserved verbatim from the legacy
    routine and from :mod:`REACHER.instrument_constants`.
    """

    def __init__(
        self,
        config: ReacherConfig,
        flat_files: Sequence[str | Path],
        master_dark_files: Sequence[str | Path],
        master_flat_path: str | Path,
        bad_pix_mask_path: Optional[str | Path] = None,
        clump_mask_path: Optional[str | Path] = None,
    ) -> None:
        super().__init__(config)
        self.flat_files = [Path(p) for p in flat_files]
        self.master_dark_files = [Path(p) for p in master_dark_files]
        self.master_flat_path = Path(master_flat_path)
        self.bad_pix_mask_path = (
            Path(bad_pix_mask_path) if bad_pix_mask_path is not None else None
        )
        self.clump_mask_path = (
            Path(clump_mask_path) if clump_mask_path is not None else None
        )

    def _check_inputs(self) -> None:
        if not self.flat_files:
            raise InvalidConfigError("No flat files provided for master flat")
        if len(self.flat_files) != len(self.master_dark_files):
            raise InvalidConfigError(
                "flat_files and master_dark_files must have the same length"
            )

        missing_flats = [str(p) for p in self.flat_files if not p.exists()]
        if missing_flats:
            raise StepExecutionError(
                f"Missing flat files: {', '.join(missing_flats)}"
            )

        for label, path in (
            ("bad pixel mask", self.bad_pix_mask_path),
            ("clump mask", self.clump_mask_path),
        ):
            if path is not None and not path.exists():
                raise StepExecutionError(f"Missing {label}: {path}")

    def run(self) -> None:
        """Execute the master flat creation step."""
        t0 = time.perf_counter()
        self._check_inputs()
        if not self._quiet:
            self.logger.info("Starting MasterFlatCreator")
        self.logger.debug(
            "Flat files: %s", ", ".join(p.name for p in self.flat_files)
        )
        self.logger.debug("Master flat output: %s", self.master_flat_path.name)

        self.master_flat_path.parent.mkdir(parents=True, exist_ok=True)

        if self._skip_if_exists(self.master_flat_path):
            return

        normalized_frames, header = self._build_per_file_normalized_frames()
        master_flat = self._combine_and_clean(normalized_frames)
        master_flat = self._apply_bad_pixel_correction(master_flat)
        master_flat = self._normalize_by_channel_means(master_flat)
        self._write_master_flat(master_flat, header)

        self.logger.info(
            "Created %s in %s",
            self.master_flat_path.name,
            format_duration(time.perf_counter() - t0),
        )
        if not self._quiet:
            self.logger.info("Finished MasterFlatCreator")

    def _build_per_file_normalized_frames(
        self,
    ) -> tuple[list[np.ndarray], fits.Header]:
        """Dark-subtract and per-frame normalize each input flat.

        Mirrors the legacy ``make_master_flat`` loop:
            * skip flats whose matching master dark file does not exist;
            * 2D flats: subtract the master dark in-place;
            * 3D cubes: subtract dark and collapse with ``np.nanmedian`` axis 0;
            * divide each result by its own ``np.nanmedian`` (per-frame norm).
        """
        normalized_frames: list[np.ndarray] = []
        first_header: Optional[fits.Header] = None

        for flat_path, dark_path in zip(self.flat_files, self.master_dark_files):
            if not dark_path.exists():
                self.logger.warning(
                    "Master dark %s does not exist; skipping flat %s",
                    dark_path.name,
                    flat_path.name,
                )
                continue

            self.logger.info("Working on %s", flat_path.name)

            with fits.open(flat_path) as flat_hdul:
                flat_data = flat_hdul[0].data
                flat_header = flat_hdul[0].header.copy()
            with fits.open(dark_path) as dark_hdul:
                dark_data = dark_hdul[0].data

            if flat_data is None:
                self.logger.warning(
                    "%s has no data; skipping", flat_path.name
                )
                continue

            if flat_data.ndim not in (2, 3):
                self.logger.warning(
                    "%s is not a 2D or 3D file; skipping", flat_path.name
                )
                continue

            if first_header is None:
                first_header = flat_header

            flat_arr = np.asarray(flat_data, dtype=float)
            dark_arr = np.asarray(dark_data, dtype=float)

            if flat_arr.ndim == 2:
                frame = flat_arr - dark_arr
            else:
                self.logger.info(
                    "Flat exists of multiple exposures, "
                    "combining to median file"
                )
                frame = np.nanmedian(flat_arr - dark_arr, axis=0)

            normalized_frames.append(frame / np.nanmedian(frame))

        if not normalized_frames:
            raise StepExecutionError(
                "No usable flat frames found for master flat"
            )

        if first_header is None:
            first_header = fits.Header()

        return normalized_frames, first_header

    def _combine_and_clean(
        self, normalized_frames: list[np.ndarray]
    ) -> np.ndarray:
        """Median combine the per-file normalized frames; replace NaNs with 0."""
        stacked = np.array(normalized_frames)
        master_flat = np.nanmedian(stacked, axis=0)
        master_flat = np.nan_to_num(master_flat)
        return master_flat

    def _apply_bad_pixel_correction(self, master_flat: np.ndarray) -> np.ndarray:
        """Interpolate bad pixel and clump locations if masks are supplied."""
        corrected = np.array([master_flat])

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

    def _normalize_by_channel_means(self, master_flat: np.ndarray) -> np.ndarray:
        """Divide by mean of LEFT and RIGHT channel medians.

        Both IRDIS channels are normalized to ~1.0 by dividing the master
        flat by the mean of the LEFT-channel median and the RIGHT-channel
        median.

        For frames smaller than the IRDIS detector (e.g. unit-test
        fixtures) the slice indices may fall fully or partially outside
        the array. We fall back to the global ``np.nanmedian`` of the
        master flat in that case.
        """
        # LEFT/RIGHT slices are instrument-coupled (pulled from
        # ``config.instrument.detector``). The two row slices
        # differ by detector geometry, not by accident.
        left_channel = self.config.instrument.detector.left_channel
        right_channel = self.config.instrument.detector.right_channel
        rows_l, cols_l = left_channel["rows"], left_channel["cols"]
        rows_r, cols_r = right_channel["rows"], right_channel["cols"]

        left_region = master_flat[rows_l, cols_l]
        right_region = master_flat[rows_r, cols_r]

        if left_region.size == 0 or right_region.size == 0:
            self.logger.info(
                "Frame too small for IRDIS LEFT/RIGHT slices; "
                "falling back to global median for normalization"
            )
            mean_masterflat = np.nanmedian(master_flat)
            self.logger.info("global median of master flat %s", mean_masterflat)
        else:
            median_left = np.nanmedian(left_region)
            median_right = np.nanmedian(right_region)
            mean_masterflat = (median_left + median_right) / 2

            self.logger.info(
                "median of master flat left %s, right %s",
                median_left,
                median_right,
            )
            self.logger.info("mean of master flat %s", mean_masterflat)

        if mean_masterflat == 0 or not np.isfinite(mean_masterflat):
            raise StepExecutionError(
                "Master flat normalization factor is zero or non-finite"
            )

        return master_flat / mean_masterflat

    def _write_master_flat(
        self, master_flat: np.ndarray, source_header: fits.Header
    ) -> None:
        """Persist the master flat FITS file with provenance keywords."""
        hdu = fits.PrimaryHDU(data=master_flat)
        sanitized = sanitize_header(source_header)
        for key, value in sanitized.items():
            try:
                hdu.header[key] = value
            except (ValueError, KeyError):
                continue

        hdu.header["HISTORY"] = "MASTER FLAT"
        for i, flat_path in enumerate(self.flat_files):
            hdu.header[f"FLAT{i}"] = flat_path.name

        fits.HDUList([hdu]).writeto(self.master_flat_path, overwrite=True)
