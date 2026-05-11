"""Master sky creation step.

Refactors :func:`REACHER.routines.make_mastersky`. The master sky is a
median-combined sky frame that is later temperature- and exposure-time
scaled when subtracted from science frames in the cube assembly step.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from astropy.io import fits

from ..config import ReacherConfig
from ..exceptions import (
    HeaderKeyError,
    InvalidConfigError,
    StepExecutionError,
)
from ..io.fits_handler import sanitize_header
from .base import PipelineStep


class MasterSkyCreator(PipelineStep):
    """Build a master sky frame from a list of sky observations.

    The algorithm preserves :func:`routines.make_mastersky`:

    1. If ``master_sky_path`` already exists and ``overwrite=False`` the
       step is skipped (resumability).
    2. For each sky file, read the data and the temperature keyword
       configured via ``config.instrument.keys.detector_temp``
       (defaults to the SPHERE ``HIERARCH ESO INS4 TEMP449 VAL``).
        * 2D arrays are appended as-is.
        * 3D cubes are collapsed via ``np.nanmedian`` along axis 0.
    3. The per-file frames are stacked and median combined along axis 0.
    4. The result is written with a sanitized header plus the provenance
       keywords:

       * ``MEANTEMP`` -- mean of per-file temperatures (used by
         :class:`CubeAssembler` and :class:`FluxCalibrator` to
         temperature-scale the sky subtraction).
       * ``SKYTMP0..N`` -- the per-file temperatures.
       * ``SKY0..N`` -- per-file basenames.
       * ``HISTORY = MASTER SKY``.
       * ``EXPTIME`` -- preserved from the first sky file's header (the
         downstream sky-subtraction code reads ``EXPTIME`` of the master
         sky).

    Parameters
    ----------
    config
        Reacher configuration object.
    sky_files
        Sequence of input sky FITS file paths.
    master_sky_path
        Output FITS path for the master sky.

    Raises
    ------
    InvalidConfigError
        If ``sky_files`` is empty.
    StepExecutionError
        If any input file is missing, contains no data, has unsupported
        dimensionality, or if no usable frames are produced.
    HeaderKeyError
        If the temperature keyword
        (``config.instrument.keys.detector_temp``) is missing in any
        sky file.

    Notes
    -----
    Behavior changes vs legacy (per Architecture Rule 7):

    * The legacy keyword name was ``owerwrite`` (typo); the step uses
      :attr:`config.processing.overwrite` instead.
    * The legacy routine wrote ``SKYTEMP`` in a loop, which silently
      overwrites the keyword and only retains the **last** sky
      temperature. We instead emit numbered ``SKYTMP0..N`` keywords
      (matching the ``DARK0..N`` / ``FLAT0..N`` provenance style),
      preserving information that the legacy code dropped.
    """

    def __init__(
        self,
        config: ReacherConfig,
        sky_files: Sequence[str | Path],
        master_sky_path: str | Path,
    ) -> None:
        super().__init__(config)
        self.sky_files = [Path(p) for p in sky_files]
        self.master_sky_path = Path(master_sky_path)
        # Detector-temperature header keyword: instrument-coupled
        # (read from ``config.instrument.keys``).
        self._temp_key = config.instrument.keys.detector_temp

    def _check_inputs(self) -> None:
        if not self.sky_files:
            raise InvalidConfigError("No sky files provided for master sky")
        missing = [str(p) for p in self.sky_files if not p.exists()]
        if missing:
            raise StepExecutionError(
                f"Missing sky files: {', '.join(missing)}"
            )

    def run(self) -> None:
        """Execute the master sky creation step."""
        self._check_inputs()
        if not self._quiet:
            self.logger.info("Starting MasterSkyCreator")
        self.logger.debug(
            "Sky files: %s", ", ".join(p.name for p in self.sky_files)
        )
        self.logger.debug("Master sky output: %s", self.master_sky_path.name)

        self.master_sky_path.parent.mkdir(parents=True, exist_ok=True)

        if self._skip_if_exists(self.master_sky_path):
            return

        sky_frames, sky_temps, header = self._load_sky_frames()
        master_sky = np.nanmedian(np.array(sky_frames), axis=0)
        self._write_master_sky(master_sky, sky_temps, header)

        self.logger.debug(
            "created master sky %s", self.master_sky_path.name
        )
        if not self._quiet:
            self.logger.info("Finished MasterSkyCreator")

    def _load_sky_frames(
        self,
    ) -> tuple[list[np.ndarray], list[float], fits.Header]:
        """Load each sky file and collect its frame + temperature."""
        sky_frames: list[np.ndarray] = []
        sky_temps: list[float] = []
        first_header: Optional[fits.Header] = None

        for sky_path in self.sky_files:
            with fits.open(sky_path) as hdul:
                data = hdul[0].data
                header = hdul[0].header.copy()

            if data is None:
                raise StepExecutionError(
                    f"Sky file {sky_path.name} contains no data"
                )

            if self._temp_key not in header:
                raise HeaderKeyError(
                    f"Sky file {sky_path.name} is missing keyword "
                    f"{self._temp_key!r}"
                )

            sky_temps.append(float(header[self._temp_key]))

            if first_header is None:
                first_header = header

            if data.ndim == 2:
                sky_frames.append(np.asarray(data, dtype=float))
            elif data.ndim == 3:
                self.logger.info(
                    "Sky data exists of multiple exposures, "
                    "combining to median file"
                )
                sky_frames.append(
                    np.nanmedian(np.asarray(data, dtype=float), axis=0)
                )
            else:
                raise StepExecutionError(
                    f"Sky file {sky_path.name} has unsupported ndim="
                    f"{data.ndim}"
                )

        if not sky_frames:
            raise StepExecutionError(
                "No usable sky frames found for master sky"
            )

        if first_header is None:
            first_header = fits.Header()

        return sky_frames, sky_temps, first_header

    def _write_master_sky(
        self,
        master_sky: np.ndarray,
        sky_temps: Sequence[float],
        source_header: fits.Header,
    ) -> None:
        """Persist the master sky FITS file with provenance keywords."""
        hdu = fits.PrimaryHDU(data=master_sky)

        sanitized = sanitize_header(source_header)
        for key, value in sanitized.items():
            try:
                hdu.header[key] = value
            except (ValueError, KeyError):
                continue

        hdu.header["MEANTEMP"] = float(np.mean(sky_temps))
        for i, temp in enumerate(sky_temps):
            hdu.header[f"SKYTMP{i}"] = float(temp)
        for i, sky_path in enumerate(self.sky_files):
            hdu.header[f"SKY{i}"] = sky_path.name

        hdu.header["HISTORY"] = "MASTER SKY"

        fits.HDUList([hdu]).writeto(self.master_sky_path, overwrite=True)
