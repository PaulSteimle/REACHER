"""Flux calibration step.

Refactor of :func:`REACHER.routines.create_flux_file`. Builds a flux
reference frame for a single flux observation:

1. Median-stack the input cube to a 2D image.
2. (Optional) subtract a temperature/exposure-scaled master sky.
3. Fit a 2D Gaussian to find the on-sky position of the calibration star.
4. Shift and crop the image so the star is at the center of a small
   ``cut_out_size`` postage stamp.
5. Subtract a sky-background estimate measured from an annulus around the
   star.
6. Rescale by ``exp_time_ratio / nd_ratio`` so the flux is normalized to
   the science exposure time and the ND filter throughput.

The output is written as a small FITS file with ``EXPTIMERATIO``,
``ND_RATIO`` and ``BACKGROUND`` provenance keywords.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from astropy.io import fits

from ..config import ReacherConfig
from ..exceptions import (
    HeaderKeyError,
    InvalidConfigError,
    StepExecutionError,
)
from ..io.fits_handler import sanitize_header
from ..utils.filters import get_ratio_for_nd_filter
from .base import PipelineStep




class FluxCalibrator(PipelineStep):
    """Build a flux calibration reference frame for one observation.

    Parameters
    ----------
    config
        Reacher configuration object. ``processing.make_plots`` controls
        plot generation; ``processing.overwrite`` controls resumability;
        ``processing.save_fullsize_flux`` controls the optional fullsize
        FITS dump; ``calibration.flux_cut_out_size`` is the postage-stamp
        size; ``calibration.star_loc`` is the initial Gaussian-fit
        center; ``calibration.image_shift_library`` and
        ``calibration.image_shift_interpolation`` configure the sub-pixel
        shift.
    input_file
        Path to the input flux FITS cube.
    output_file
        Path where the (cropped) flux file will be written.
    science_exp_time
        Exposure time of the matching science observation (used to
        compute ``exp_time_ratio``).
    channel
        ``'LEFT'`` or ``'RIGHT'`` IRDIS channel.
    master_sky_path
        Optional path to a master sky FITS file. When ``None`` the
        sky-subtraction step is skipped (matches the legacy
        ``sky_files_exist=False`` branch).

    Raises
    ------
    InvalidConfigError
        If a required argument is missing or the channel is not
        ``LEFT``/``RIGHT``.
    StepExecutionError
        If the input file is missing, the data is unusable, or the
        Gaussian fit returns NaN coordinates.
    HeaderKeyError
        If a required header keyword is missing from the input file.

    Notes
    -----
    Behavior changes vs legacy (per Architecture Rule 7):

    * The legacy ``create_flux_file`` referenced an undefined
      ``rescaled_image`` variable inside the NaN-center fallback
      plotting block (a use-before-assignment bug). The fallback now
      raises :class:`StepExecutionError` so the user sees a clear
      diagnostic instead of a confusing ``NameError``. This preserves
      the surface contract for the happy path; only the bug branch is
      changed.
    * Plot generation is gated on ``config.processing.make_plots``.
    * The legacy ``cut_out_size%2 == 1`` increment branch is preserved
      exactly (the comment claims ``cut_out_size`` should be odd but the
      condition increments when it already is — this is a known legacy
      quirk that we surface as a warning instead of silently fixing).
    """

    def __init__(
        self,
        config: ReacherConfig,
        input_file: str | Path,
        output_file: str | Path,
        science_exp_time: float,
        channel: str,
        master_sky_path: Optional[str | Path] = None,
    ) -> None:
        super().__init__(config)
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.science_exp_time = float(science_exp_time)
        if channel not in ("LEFT", "RIGHT"):
            raise InvalidConfigError(
                f"channel must be 'LEFT' or 'RIGHT'; got {channel!r}"
            )
        self.channel = channel
        self.master_sky_path = (
            Path(master_sky_path) if master_sky_path is not None else None
        )
        # Instrument-coupled header keywords (read from
        # ``config.instrument.keys``).
        self._temp_key = config.instrument.keys.detector_temp
        self._ndfilter_key = config.instrument.keys.nd_filter_id
        self._wavefilter_key = config.instrument.keys.filter_name

    def _check_inputs(self) -> None:
        if not self.input_file.exists():
            raise StepExecutionError(
                f"Missing input flux file: {self.input_file}"
            )
        if self.master_sky_path is not None and not self.master_sky_path.exists():
            raise StepExecutionError(
                f"Missing master sky file: {self.master_sky_path}"
            )
        if self.science_exp_time <= 0:
            raise InvalidConfigError(
                f"science_exp_time must be > 0; got {self.science_exp_time}"
            )

    def run(self) -> None:
        """Execute the flux calibration step."""
        self._check_inputs()
        if not self._quiet:
            self.logger.info("Starting FluxCalibrator")
        self.logger.debug(
            "Input: %s -> output: %s",
            self.input_file.name,
            self.output_file.name,
        )

        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        if self._skip_if_exists(self.output_file):
            return

        with fits.open(self.input_file) as hdul:
            data = hdul[0].data
            header = hdul[0].header.copy()

        if data is None or data.ndim != 3:
            raise StepExecutionError(
                f"Flux file {self.input_file.name} must be a 3D cube; "
                f"got {None if data is None else data.shape}"
            )

        for key in (self._temp_key, self._ndfilter_key, self._wavefilter_key, "EXPTIME"):
            if key not in header:
                raise HeaderKeyError(
                    f"Flux file {self.input_file.name} missing keyword {key!r}"
                )

        exp_time = float(header["EXPTIME"])
        temp = float(header[self._temp_key])
        nd_filter = header[self._ndfilter_key]
        wave_filter = header[self._wavefilter_key]

        nd_ratio = get_ratio_for_nd_filter(
            wave_filter=wave_filter,
            nd_filter=nd_filter,
            channel=self.channel,
        )
        self.logger.info(
            "nd filter ratio for flux file %s: %s",
            self.output_file.name, nd_ratio,
        )

        exp_time_ratio = self.science_exp_time / exp_time
        self.logger.info(
            "exp time ratio for flux file %s: %s",
            self.output_file.name, exp_time_ratio,
        )

        stacked = np.nanmedian(np.nan_to_num(data), axis=0)

        if self.master_sky_path is not None:
            stacked = self._subtract_sky(stacked, temp, exp_time)

        med_y, med_x = self._fit_center(stacked)

        cut_out_size = self._resolve_cut_out_size(stacked)
        cut_out_half = int(np.floor((cut_out_size - 1) / 2))

        int_center_x = int(np.floor(med_x))
        int_center_y = int(np.floor(med_y))
        float_center_x = med_x - int_center_x
        float_center_y = med_y - int_center_y

        float_centered = self._frame_shift(
            stacked, -float_center_y, -float_center_x
        )

        # Cut out cut_out_size around the (now full-pixel-centered) star.
        int_centered = float_centered[
            int_center_y - cut_out_half : int_center_y + cut_out_half + 1,
            int_center_x - cut_out_half : int_center_x + cut_out_half + 1,
        ]

        med_background = self._estimate_background(
            float_centered, int_center_y, int_center_x
        )
        self.logger.info(
            "median sky background in file %s: %s",
            self.output_file.name, med_background,
        )

        int_centered = int_centered - med_background

        # Apply ND + exposure-time normalization.
        rescaled_image = int_centered * exp_time_ratio / nd_ratio

        if self.config.processing.make_plots:
            self._make_plots(
                rescaled_image=rescaled_image,
                full_image=stacked * exp_time_ratio / nd_ratio,
                med_x=med_x, med_y=med_y,
            )

        self._write_output(
            rescaled_image=rescaled_image,
            stacked=stacked,
            header=header,
            exp_time_ratio=exp_time_ratio,
            nd_ratio=nd_ratio,
            med_background=med_background,
        )

        if not self._quiet:
            self.logger.info("Finished FluxCalibrator")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _subtract_sky(
        self, stacked: np.ndarray, temp: float, exp_time: float
    ) -> np.ndarray:
        """Subtract the temperature- and exposure-time-scaled master sky.

        ``master_sky * (T/T_mean) * (t_frame/t_sky)`` — preserves the
        legacy scaling expression.
        """
        try:
            master_sky = fits.getdata(self.master_sky_path)
            sky_header = fits.getheader(self.master_sky_path)
            mean_sky_temp = float(sky_header["MEANTEMP"])
            sky_exptime = float(sky_header["EXPTIME"])
        except Exception as exc:
            raise StepExecutionError(
                f"Could not load master sky {self.master_sky_path.name}: "
                f"{exc}"
            ) from exc

        return (
            stacked
            - master_sky * temp / mean_sky_temp * exp_time / sky_exptime
        )

    def _resolve_cut_out_size(self, stacked: np.ndarray) -> int:
        """Resolve the postage-stamp size, preserving the legacy quirk.

        The legacy code did:

        .. code-block:: python

            if len(stacked_flux_image)%2 == 1:
                cropsize = 31
            elif len(stacked_flux_image)%2 == 0:
                cropsize = 30

        before fitting the Gaussian. Then later it took
        ``cut_out_size`` from configuration and incremented it if odd
        (the comment says ``cut_out_size should be odd``; the
        increment is the opposite of the comment — known legacy quirk).
        We preserve both behaviors.
        """
        cut_out_size = int(self.config.calibration.flux_cut_out_size)
        if cut_out_size % 2 == 1:
            self.logger.info(
                "cut_out_size should be odd, increasing by 1 (legacy quirk)"
            )
            cut_out_size += 1
        return cut_out_size

    def _fit_center(self, stacked: np.ndarray) -> Tuple[float, float]:
        """Fit a 2D Gaussian around ``calibration.star_loc``.

        Defers the import of VIP-HCI's :func:`fit_2dgaussian` so that
        this step file can be imported without the heavy science deps
        being available. The legacy routine retried with ``debug=True``
        on NaN; we replicate that retry but raise a clear
        :class:`StepExecutionError` if the fit still fails (this is
        where the legacy code referenced the undefined
        ``rescaled_image`` variable).
        """
        from vip_hci.var import fit_2dgaussian

        cropsize = 31 if len(stacked) % 2 == 1 else 30
        star_loc = tuple(self.config.calibration.star_loc)

        def _try_fit(debug_flag: bool):
            try:
                return fit_2dgaussian(
                    stacked, crop=True, cent=star_loc, cropsize=cropsize,
                    threshold=True, sigfactor=4,
                    full_output=False, debug=debug_flag,
                )
            except Exception as exc:  # numerical fit failures, NaN/Inf etc.
                self.logger.warning(
                    "Gaussian fit raised %s: %s",
                    type(exc).__name__, exc,
                )
                return float("nan"), float("nan")

        med_y, med_x = _try_fit(debug_flag=False)

        if np.isnan(med_y) or np.isnan(med_x):
            self.logger.warning(
                "Gaussian fit returned NaN; retrying with debug=True"
            )
            med_y, med_x = _try_fit(debug_flag=True)

        if np.isnan(med_y) or np.isnan(med_x):
            raise StepExecutionError(
                f"Gaussian center fit failed for {self.input_file.name} "
                f"(returned NaN / fit error). The legacy routine had a "
                f"use-before-assignment bug here; the step now stops "
                f"with this error instead."
            )

        return float(med_y), float(med_x)

    def _frame_shift(
        self, image: np.ndarray, dy: float, dx: float
    ) -> np.ndarray:
        """Sub-pixel shift via VIP's :func:`frame_shift`."""
        from vip_hci.preproc import frame_shift

        return frame_shift(
            image, dy, dx,
            imlib=self.config.calibration.image_shift_library,
            interpolation=self.config.calibration.image_shift_interpolation,
        )

    def _estimate_background(
        self, image: np.ndarray, cy: int, cx: int
    ) -> float:
        """Estimate the median background in a 301x301 annulus around the star.

        The legacy uses a 301x301 region with the central 100x100 region
        masked out as NaN; we preserve that exactly.
        """
        bg = image.copy()
        bg = bg[cy - 150 : cy + 151, cx - 150 : cx + 151]
        if bg.size == 0:
            self.logger.warning(
                "Background annulus is empty (star too close to edge); "
                "falling back to global nanmedian"
            )
            return float(np.nanmedian(image))
        bg[100:200, 100:200] = np.nan
        return float(np.nanmedian(bg))

    def _make_plots(
        self,
        rescaled_image: np.ndarray,
        full_image: np.ndarray,
        med_x: float,
        med_y: float,
    ) -> None:
        """Render the per-flux quick-look PDFs (gated on make_plots)."""
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        from ..plots import show_img

        plot_dir = self.output_file.parent / "flux_pngs"
        plot_dir.mkdir(parents=True, exist_ok=True)

        try:
            show_img(rescaled_image, minper=1, maxper=99, size=5)
            plt.scatter(20, 20, marker="x", color="red", s=100)
            plt.savefig(
                plot_dir / self.output_file.name.replace(".fits", ".pdf"),
                bbox_inches="tight", dpi=120,
            )
            plt.close()

            show_img(full_image, minper=1, maxper=99, size=5)
            plt.scatter(med_x, med_y, marker="x", color="red", s=100)
            plt.xlim(med_x - 20, med_x + 20)
            plt.ylim(med_y - 20, med_y + 20)
            plt.savefig(
                plot_dir / self.output_file.name.replace(".fits", "2.pdf"),
                bbox_inches="tight", dpi=120,
            )
            plt.close()
        except Exception as exc:  # pragma: no cover
            self.logger.warning("flux plot rendering failed: %s", exc)

    def _write_output(
        self,
        rescaled_image: np.ndarray,
        stacked: np.ndarray,
        header: fits.Header,
        exp_time_ratio: float,
        nd_ratio: float,
        med_background: float,
    ) -> None:
        """Persist the cropped flux file and (optionally) a full-size copy."""
        hdu = fits.PrimaryHDU(data=rescaled_image)
        sanitized = sanitize_header(header)
        for key, value in sanitized.items():
            try:
                hdu.header[key] = value
            except (ValueError, KeyError):
                continue

        hdu.header["EXPTIMERATIO"] = exp_time_ratio
        hdu.header["ND_RATIO"] = 1.0 / nd_ratio
        hdu.header["BACKGROUND"] = med_background

        fits.HDUList([hdu]).writeto(self.output_file, overwrite=True)

        if self.config.processing.save_fullsize_flux:
            fullsize_path = Path(
                str(self.output_file).replace(".fits", "_fullsize.fits")
            )
            full_data = stacked * exp_time_ratio / nd_ratio
            full_hdu = fits.PrimaryHDU(data=full_data)
            for key, value in sanitized.items():
                try:
                    full_hdu.header[key] = value
                except (ValueError, KeyError):
                    continue
            full_hdu.header["EXPTIMERATIO"] = exp_time_ratio
            full_hdu.header["ND_RATIO"] = 1.0 / nd_ratio
            fits.HDUList([full_hdu]).writeto(fullsize_path, overwrite=True)
