"""Cube assembly step.

Refactor of :func:`REACHER.routines.make_the_cube`. Combines a list of
calibrated science exposures into a single master cube that is the input
of the post-processing (ADI/SDI) stages. The final products are:

* ``output_file_cube`` -- the master science cube (FITS).
* ``output_file_angs`` -- the per-frame parallactic angles (text).
* ``header_infos.csv`` and ``times.csv`` -- per-file metadata (CSV).

The pipeline preserves the legacy algorithm step by step:

1. For each science file: collect parallactic angles, dither shifts,
   filter temperature, exposure time, and per-file frames.
2. Save angles + bookkeeping CSVs.
3. (Optional) Master sky subtraction with temperature + exposure-time
   scaling (``sky * T/T_mean * t/t_sky``).
4. Apply integer dither correction via ``np.roll``.
5. Pad the cube to ``1025x1025`` with the integer part of the
   center shift applied.
6. (Optional) Cosmic-ray correction (frame-by-frame, multiprocessing).
7. Sub-pixel center shift via :func:`vip_hci.preproc.cube_shift`.
8. (Optional) Anamorphic distortion correction (always uses opencv +
   lanczos4 because vip-fft does not support non-equal axis scaling).
9. (Optional) DFT upsampling dither correction with
   :func:`vip_hci.preproc.cube_recenter_dft_upsampling`.
10. Apply NaN mask + circular center mask + (optional) per-frame
    median sky background subtraction.
11. Save final cube + provenance keywords.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time

from ..config import ReacherConfig
from ..exceptions import (
    HeaderKeyError,
    InvalidConfigError,
    ShapeMismatchError,
    StepExecutionError,
)
from ..utils.cosmics import correct_cosmic_rays_cube
from .base import PipelineStep


# DATE-OBS / DATE / EXPTIME are FITS-standard keywords and are read
# directly. The other four header keywords are instrument-coupled and
# are pulled from ``config.instrument.keys`` at run time.
_FITS_STANDARD_HEADER_KEYS = ("DATE-OBS", "DATE", "EXPTIME")

_PADDED_SHAPE = (1025, 1025)


def _circular_aperture(shape, radius, inverted=False):
    """Inline circular aperture mask helper.

    Mirrors :func:`REACHER.misc.circular_aperture` to avoid a circular
    import via ``REACHER.plots`` / VIP.
    """
    yy, xx = np.indices(shape)
    cy = (shape[0] - 1) / 2.0
    cx = (shape[1] - 1) / 2.0
    distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    if inverted:
        return (distance > radius).astype(float)
    return (distance <= radius).astype(float)


class CubeAssembler(PipelineStep):
    """Assemble per-file calibrated science FITS into the master cube.

    Parameters
    ----------
    config
        Reacher configuration object. Reads ``processing``,
        ``calibration`` and ``instrument`` sub-configs.
    science_files
        Sequence of input calibrated science FITS files (in time order).
    output_file_cube
        Path where the master cube FITS will be written.
    output_file_angs
        Path where the per-frame parallactic angles will be written
        (text, one angle per line).
    shift
        ``(center_shifty, center_shiftx)`` -- the satellite-spot derived
        center shift to apply to every frame. (Computed by an external
        centering step; kept as input for now.)
    star_center_file
        Basename / path of the file from which the centering shift was
        derived. Stored in the output FITS header for provenance.
    star_name
        Name of the science target (used for plot titles).
    obs_night
        Observation night identifier (used for plot titles).
    filter_name
        IRDIS filter name (used for plot titles).
    master_sky_path
        Optional path to a master sky FITS file. ``None`` skips sky
        subtraction.

    Raises
    ------
    InvalidConfigError
        If a required input is missing.
    StepExecutionError
        If any input file is unreadable, has a missing header keyword,
        or the dither shifts contain NaNs.
    ShapeMismatchError
        If the resulting padded cube cannot be sliced consistently
        (e.g., odd anamorphic crop differences).

    Notes
    -----
    Behavior changes vs legacy (per Architecture Rule 7):

    * The legacy routine ended with ``masked_cube -= median_sky_bkg``
      after a per-frame median subtraction loop, which silently
      subtracted the *last* frame's median from the entire cube a
      second time. That trailing double-subtraction is a bug and is
      removed here.
    * Cosmic-ray cleaning uses the new
      :func:`REACHER.utils.cosmics.correct_cosmic_rays_cube` helper
      (numpy-only; the legacy ``clean_bp`` imported the optional
      ``bottleneck`` package which is not in the current environment).
    """

    def __init__(
        self,
        config: ReacherConfig,
        science_files: Sequence[str | Path],
        output_file_cube: str | Path,
        output_file_angs: str | Path,
        shift: Tuple[float, float],
        star_center_file: str | Path,
        star_name: str = "",
        obs_night: str = "",
        filter_name: str = "",
        master_sky_path: Optional[str | Path] = None,
    ) -> None:
        super().__init__(config)
        if not science_files:
            raise InvalidConfigError(
                "CubeAssembler requires at least one science file"
            )
        self.science_files = [Path(p) for p in science_files]
        self.output_file_cube = Path(output_file_cube)
        self.output_file_angs = Path(output_file_angs)
        self.shift = (float(shift[0]), float(shift[1]))
        self.star_center_file = Path(star_center_file)
        self.star_name = star_name
        self.obs_night = obs_night
        self.filter_name = filter_name
        self.master_sky_path = (
            Path(master_sky_path) if master_sky_path is not None else None
        )
        # Instrument-coupled header keywords (read from
        # ``config.instrument.keys``). Cached on the instance so
        # ``_load_per_file_frames`` can look them up by name.
        keys = config.instrument.keys
        self._parang_start_key = keys.parang_start
        self._parang_end_key = keys.parang_end
        self._dither_x_key = keys.dither_x
        self._dither_y_key = keys.dither_y
        self._temp_key = keys.detector_temp
        self._required_header_keys = (
            self._parang_end_key,
            self._parang_start_key,
            "DATE-OBS",
            "DATE",
            self._dither_x_key,
            self._dither_y_key,
            self._temp_key,
            "EXPTIME",
        )

    def _check_inputs(self) -> None:
        missing = [str(p) for p in self.science_files if not p.exists()]
        if missing:
            raise StepExecutionError(
                f"Missing science files: {', '.join(missing)}"
            )
        if (
            self.master_sky_path is not None
            and not self.master_sky_path.exists()
        ):
            raise StepExecutionError(
                f"Missing master sky: {self.master_sky_path}"
            )

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def run(self) -> None:
        """Execute the cube assembly step."""
        self._check_inputs()
        if not self._quiet:
            self.logger.info("Starting CubeAssembler")
        self.logger.debug(
            "Science files: %s",
            ", ".join(p.name for p in self.science_files),
        )
        self.logger.debug("Cube output: %s", self.output_file_cube.name)
        self.logger.debug("Angles output: %s", self.output_file_angs.name)

        self.output_file_cube.parent.mkdir(parents=True, exist_ok=True)
        self.output_file_angs.parent.mkdir(parents=True, exist_ok=True)

        if self._skip_if_exists(self.output_file_cube):
            return

        cube, angs, header_infos, times_df = self._load_per_file_frames()

        # Save the angles + bookkeeping CSVs early so they exist even
        # when do_cubes=False (matches legacy behavior).
        np.savetxt(self.output_file_angs, angs, fmt="%f")
        header_infos.to_csv(
            self.output_file_cube.parent / "header_infos.csv", index=False
        )
        times_df.to_csv(
            self.output_file_cube.parent / "times.csv", index=False
        )

        if not self.config.processing.do_cubes:
            self.logger.info("not making the cube, returning!")
            return

        per_frame = self._expand_per_frame_metadata(header_infos, len(cube))
        cube = self._optional_sky_subtract(cube, per_frame)
        cube = self._apply_integer_dither(cube, per_frame)

        center_shifty, center_shiftx = self.shift
        int_shift_y = int(np.floor(center_shifty))
        int_shift_x = int(np.floor(center_shiftx))
        float_shift_y = center_shifty - int_shift_y
        float_shift_x = center_shiftx - int_shift_x

        expanded_cube, nan_mask = self._pad_and_integer_center(
            cube, int_shift_y, int_shift_x
        )

        if not self.config.processing.skip_cosmics:
            self.logger.info("correcting the cube for cosmic rays")
            cube_sigcorr = correct_cosmic_rays_cube(
                np.nan_to_num(expanded_cube),
                n_sigma=self.config.processing.n_sigma,
                box_radius=self.config.processing.box_radius,
                nproc=self.config.processing.nproc,
            )
        else:
            cube_sigcorr = expanded_cube

        float_centered_cube = self._sub_pixel_center(
            cube_sigcorr, float_shift_y, float_shift_x
        )

        anam_corr_cube = self._optional_anamorphic_correction(
            float_centered_cube
        )

        dither_corr_cube = self._optional_dft_dither_correction(
            anam_corr_cube, cube
        )

        # NaN mask + central exclusion (legacy circular_aperture-based)
        masked_cube = dither_corr_cube * nan_mask
        center_mask = _circular_aperture(masked_cube[0].shape, 300, inverted=True)

        if self.config.processing.subtract_median_science_sky_bkg:
            masked_cube = self._per_frame_sky_bkg_subtract(
                masked_cube, center_mask
            )

        self._write_cube(masked_cube)
        if not self._quiet:
            self.logger.info("cube finished!")


    # ------------------------------------------------------------------
    # Stage-1: per-file load
    # ------------------------------------------------------------------
    def _load_per_file_frames(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
        """Concatenate all science files into one cube + per-frame angle list."""
        sphere_offset = self.config.instrument.sphere_pa_offset
        true_north = self.config.instrument.sphere_true_north_correction

        frames: list[np.ndarray] = []
        angs: list[float] = []
        all_times: list[str] = []
        all_times_mjd: list[float] = []
        rows: list[dict] = []

        self.logger.info(
            "extracting the single exposures from the files into one cube"
        )
        for sci_path in self.science_files:
            with fits.open(sci_path) as hdul:
                data = hdul[0].data
                header = hdul[0].header

            if data is None:
                raise StepExecutionError(
                    f"Science file {sci_path.name} contains no data"
                )

            for key in self._required_header_keys:
                if key not in header:
                    raise HeaderKeyError(
                        f"{sci_path.name} missing keyword {key!r}"
                    )

            angend = float(header[self._parang_end_key])
            angstart = float(header[self._parang_start_key])
            time_start = header["DATE-OBS"]
            time_end = header["DATE"]
            dith_x = float(header[self._dither_x_key])
            dith_y = float(header[self._dither_y_key])
            temp = float(header[self._temp_key])
            exp_time = float(header["EXPTIME"])

            n_frames = len(data) if data.ndim == 3 else 1

            self.logger.info(
                "file %s has %s frames from angle %s to %s deg",
                sci_path.name, n_frames, angstart, angend,
            )

            rows.append({
                "file": sci_path.name,
                "start_angle": angstart,
                "end_angle": angend,
                "dither_x": dith_x,
                "dither_y": dith_y,
                "n_frames": n_frames,
                "filter_temp": temp,
                "exp_time": exp_time,
            })

            # 180-deg crossing correction (preserve legacy logic exactly).
            if angend * angstart < 0 and abs(angend) + abs(angstart) > 180:
                if angstart < 0:
                    angstart = angstart % 360
                elif angend < 0:
                    angend = angend % 360

            angs_i = (
                angstart
                + (angend - angstart) * (0.5 + np.arange(n_frames)) / n_frames
            )

            start_mjd = Time(time_start, format="isot", scale="utc").mjd
            end_mjd = Time(time_end, format="isot", scale="utc").mjd
            times_mjd = np.linspace(start_mjd, end_mjd, n_frames)
            times = Time(times_mjd, format="mjd").iso

            angs_i = (angs_i + sphere_offset + true_north) % 360

            data_iter = data if data.ndim == 3 else np.array([data])
            for ang_v, image, t, t_mjd in zip(
                angs_i, data_iter, times, times_mjd
            ):
                frames.append(image)
                angs.append(float(ang_v))
                all_times.append(str(t))
                all_times_mjd.append(float(t_mjd))

        if not frames:
            raise StepExecutionError("No frames loaded from science files")

        cube = np.array(frames)
        angs_arr = np.array(angs)

        header_infos = pd.DataFrame(rows)
        times_df = pd.DataFrame({"time": all_times, "mjd": all_times_mjd})

        return cube, angs_arr, header_infos, times_df

    # ------------------------------------------------------------------
    # Stage-2: per-frame metadata
    # ------------------------------------------------------------------
    def _expand_per_frame_metadata(
        self, header_infos: pd.DataFrame, total_frames: int
    ) -> dict:
        """Expand per-file metadata to per-frame arrays."""
        all_x = np.full(total_frames, np.nan)
        all_y = np.full(total_frames, np.nan)
        temps = np.full(total_frames, np.nan)
        exp_times = np.full(total_frames, np.nan)

        k = 0
        for i, n_exp in enumerate(header_infos["n_frames"]):
            for _ in range(int(n_exp)):
                all_x[k] = header_infos["dither_x"][i]
                all_y[k] = header_infos["dither_y"][i]
                temps[k] = header_infos["filter_temp"][i]
                exp_times[k] = header_infos["exp_time"][i]
                k += 1

        if (
            np.isnan(all_x).any()
            or np.isnan(all_y).any()
            or np.isnan(temps).any()
            or np.isnan(exp_times).any()
        ):
            raise StepExecutionError(
                "Dither shifts / temps / exptimes contain NaNs; "
                "check the science headers"
            )

        return {
            "x_dith": all_x,
            "y_dith": all_y,
            "temps": temps,
            "exp_times": exp_times,
        }

    def _optional_sky_subtract(
        self, cube: np.ndarray, per_frame: dict
    ) -> np.ndarray:
        if self.master_sky_path is None:
            return cube

        try:
            master_sky = fits.getdata(self.master_sky_path)
            sky_header = fits.getheader(self.master_sky_path)
            mean_sky_temp = float(sky_header["MEANTEMP"])
            sky_exptime = float(sky_header["EXPTIME"])
        except Exception as exc:
            raise StepExecutionError(
                f"Could not load master sky "
                f"{self.master_sky_path.name}: {exc}"
            ) from exc

        out = cube.astype(float, copy=True)
        for i in range(len(out)):
            scale = (
                per_frame["temps"][i] / mean_sky_temp
                * per_frame["exp_times"][i] / sky_exptime
            )
            out[i] = out[i] - master_sky * scale
        return out

    def _apply_integer_dither(
        self, cube: np.ndarray, per_frame: dict
    ) -> np.ndarray:
        """Integer dither correction using ``np.roll``."""
        out = cube.copy()
        for i, (xx, yy) in enumerate(
            zip(per_frame["x_dith"], per_frame["y_dith"])
        ):
            out[i] = np.roll(
                out[i], (-int(yy), -int(xx)), axis=(0, 1)
            )
        return out

    def _pad_and_integer_center(
        self, cube: np.ndarray, int_shift_y: int, int_shift_x: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pad each frame to 1025x1025 with the integer center applied.

        Returns the padded cube plus a 2D ``nan_mask`` of the same
        shape (1.0 inside the original frame box, NaN outside) used to
        zero out the padded regions in the final output.
        """
        y_size = cube.shape[1]
        x_size = cube.shape[2]
        ph, pw = _PADDED_SHAPE
        diff_y = (ph - y_size) / 2.0
        diff_x = (pw - x_size) / 2.0

        if diff_y < 0 or diff_x < 0:
            raise ShapeMismatchError(
                f"Frame shape {(y_size, x_size)} larger than padded "
                f"shape {_PADDED_SHAPE}; cannot center"
            )

        expanded = np.zeros((len(cube), ph, pw))
        expanded[
            :,
            int(diff_y + int_shift_y) : int(-diff_y + int_shift_y) or None,
            int(diff_x + int_shift_x) : int(-diff_x + int_shift_x) or None,
        ] = cube

        nan_mask = np.zeros((ph, pw)) * np.nan
        nan_mask[
            int(diff_y + int_shift_y) : int(-diff_y + int_shift_y) or None,
            int(diff_x + int_shift_x) : int(-diff_x + int_shift_x) or None,
        ] = 1
        return expanded, nan_mask

    def _sub_pixel_center(
        self, cube: np.ndarray, dy: float, dx: float
    ) -> np.ndarray:
        from vip_hci.preproc import cube_shift

        self.logger.info("now centering the cube on a sub pixel level")
        return cube_shift(
            np.nan_to_num(cube), dy, dx,
            imlib=self.config.calibration.image_shift_library,
            interpolation=self.config.calibration.image_shift_interpolation,
        )

    # ------------------------------------------------------------------
    # Anamorphic correction
    # ------------------------------------------------------------------
    def _optional_anamorphic_correction(
        self, cube: np.ndarray
    ) -> np.ndarray:
        if not self.config.processing.correct_anamorphism:
            return cube

        scaling = self.config.instrument.sphere_anamorphic_ratio
        self.logger.info(
            "applying anamorphic scaling of %s in x and %s in y",
            scaling[0], scaling[1],
        )

        # vip-fft does not support non-equal axis scaling -> use opencv
        # + lanczos4 for the anamorphic step (Architecture Rule 11).
        if self.config.calibration.image_shift_library == "vip-fft":
            self.logger.warning(
                "vip-fft does not support non-equal axis scaling, "
                "switching to opencv - lanczos4 for the anamorphic correction"
            )
            imlib = "opencv"
            interp = "lanczos4"
        else:
            imlib = self.config.calibration.image_shift_library
            interp = self.config.calibration.image_shift_interpolation

        if cube.ndim != 3:
            raise ShapeMismatchError(
                f"Input frame must be a 3D array; got ndim={cube.ndim}"
            )

        from vip_hci.preproc import frame_px_resampling

        nproc = self.config.processing.nproc
        if nproc is None:
            nproc = mp.cpu_count() // 2 or 1

        if nproc <= 1:
            anam_cube = np.array([
                frame_px_resampling(
                    frame, scaling, imlib=imlib, interpolation=interp,
                    keep_center=True,
                )
                for frame in cube
            ])
        else:
            args = list(zip(
                [frame for frame in cube],
                [scaling] * len(cube),
                [imlib] * len(cube),
                [interp] * len(cube),
            ))
            with mp.Pool(processes=nproc) as pool:
                anam_cube = np.array(
                    pool.starmap(_anam_resample_worker, args)
                )

        return self._symmetric_crop_after_anamorphism(cube, anam_cube)

    def _symmetric_crop_after_anamorphism(
        self, before: np.ndarray, after: np.ndarray
    ) -> np.ndarray:
        """Symmetric center crop so the post-anamorphic cube matches input size."""
        x_before = before.shape[2]
        y_before = before.shape[1]
        x_after = after.shape[2]
        y_after = after.shape[1]
        x_cut = abs(x_before - x_after)
        y_cut = abs(y_before - y_after)

        if x_cut % 2 != 0 or y_cut % 2 != 0:
            raise ShapeMismatchError(
                "After distortion correction the cube size difference is "
                "odd, please check the scaling factors"
            )

        self.logger.info(
            "original cube size: x = %s, y = %s", x_before, y_before
        )
        self.logger.info(
            "anamorphic corrected cube size: x = %s, y = %s",
            x_after, y_after,
        )

        cropped = after[
            :,
            y_cut // 2 : y_after - y_cut // 2,
            x_cut // 2 : x_after - x_cut // 2,
        ]
        self.logger.info(
            "final cube size after cutting: x = %s, y = %s",
            cropped.shape[2], cropped.shape[1],
        )
        return cropped

    # ------------------------------------------------------------------
    # DFT dither correction
    # ------------------------------------------------------------------
    def _optional_dft_dither_correction(
        self, cube: np.ndarray, original_cube: np.ndarray
    ) -> np.ndarray:
        if self.config.processing.skip_dither_correction:
            return cube

        from vip_hci.preproc import cube_recenter_dft_upsampling
        from vip_hci.var import frame_center

        self.logger.info("finally correcting the cube for telescope dithers")
        out, shy2a, shx2a = cube_recenter_dft_upsampling(
            np.nan_to_num(cube),
            upsample_factor=100,
            subi_size=None,
            center_fr1=frame_center(original_cube[0]),
            fwhm=4,
            negative=True,
            full_output=True,
            debug=False,
            nproc=self.config.processing.nproc,
            imlib=self.config.calibration.image_shift_library,
            interpolation=self.config.calibration.image_shift_interpolation,
        )

        np.savetxt(
            self.output_file_cube.parent / "dither_shifts.txt",
            np.array([shy2a, shx2a]), fmt="%f",
        )
        return out

    # ------------------------------------------------------------------
    # Per-frame median sky bkg subtraction
    # ------------------------------------------------------------------
    def _per_frame_sky_bkg_subtract(
        self, cube: np.ndarray, center_mask: np.ndarray
    ) -> np.ndarray:
        """Subtract the per-frame median (computed outside ``center_mask``).

        Mirrors the legacy loop. Note the legacy had a redundant trailing
        ``masked_cube -= median_sky_bkg`` after the loop that re-applied
        the *last* frame's median to the whole cube (a bug). That line
        is intentionally removed here.
        """
        self.logger.info(
            "subtracting the median of each frame as sky background"
        )
        medians = np.full(len(cube), np.nan)
        out = cube.astype(float, copy=True)
        for i, image in enumerate(out):
            cm = image.copy()
            cm[center_mask == 0] = np.nan
            median_bkg = np.nanmedian(cm)
            out[i] = out[i] - median_bkg
            medians[i] = median_bkg

        np.savetxt(
            self.output_file_cube.parent / "median_sky_bkg.txt",
            medians, fmt="%f",
        )
        return out

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def _write_cube(self, cube: np.ndarray) -> None:
        self.logger.info(
            "saving the cube to %s", self.output_file_cube.name
        )
        hdu = fits.PrimaryHDU(data=cube)

        # Use the first science file's header as the base.
        with fits.open(self.science_files[0]) as hdul:
            header = hdul[0].header.copy()

        for i, file in enumerate(self.science_files):
            header[f"SCIENC{i}"] = file.name

        center_shifty, center_shiftx = self.shift
        header["CENTSTAR"] = self.star_center_file.name
        header["SHIFTX"] = float(np.mean(center_shiftx))
        header["SHIFTY"] = float(np.mean(center_shifty))

        if self.master_sky_path is not None:
            header["MASTSKY"] = self.master_sky_path.name
            try:
                sky_header = fits.getheader(self.master_sky_path)
                header["MNSKYTMP"] = float(sky_header["MEANTEMP"])
            except Exception:
                pass

        hdu.header = header
        fits.HDUList([hdu]).writeto(self.output_file_cube, overwrite=True)


def _anam_resample_worker(frame, scaling, imlib, interp):
    """Top-level multiprocessing worker for anamorphic correction.

    Defined at module scope so that ``multiprocessing.Pool`` (which
    requires pickle-able functions on macOS spawn semantics) can
    serialize it.
    """
    from vip_hci.preproc import frame_px_resampling

    return frame_px_resampling(
        frame, scaling, imlib=imlib, interpolation=interp, keep_center=True
    )
