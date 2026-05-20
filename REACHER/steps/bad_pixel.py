"""Bad pixel mask creation step.

The numerical helper :func:`interpolate_bad_pixels` lives in
:mod:`REACHER.utils.bad_pixel` (promoted from this module in Phase 3
because it is consumed by multiple steps). It is re-exported here for
backwards compatibility with existing tests / call sites.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from astropy.io import fits

from ..config import ReacherConfig
from ..exceptions import InvalidConfigError, StepExecutionError
from ..utils.bad_pixel import interpolate_bad_pixels  # re-exported
from ..utils.formatting import format_duration
from .base import PipelineStep

__all__ = ["BadPixelMaskCreator", "interpolate_bad_pixels"]


def _circular_aperture_mask(
    shape: Tuple[int, int], radius: int, center: Tuple[int, int]
) -> np.ndarray:
    """Create a binary circular aperture mask."""
    mask = np.zeros(shape, dtype=float)
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    distance = np.sqrt((xx - center[1]) ** 2 + (yy - center[0]) ** 2)
    mask[distance <= radius] = 1.0
    return mask


class BadPixelMaskCreator(PipelineStep):
    """Create bad pixel and clump masks from dark frames.

    Parameters
    ----------
    config
        Reacher configuration object.
    dark_files
        List of dark file paths.
    bad_pix_mask_path
        Output path for the bad pixel mask.
    clump_mask_path
        Output path for the fixed IRDIS clump mask.
    coef
        Sigma clipping coefficient for the MAD threshold.
    """

    def __init__(
        self,
        config: ReacherConfig,
        dark_files: Sequence[str | Path],
        bad_pix_mask_path: str | Path,
        clump_mask_path: str | Path,
        coef: float = 5.0,
    ) -> None:
        super().__init__(config)
        self.dark_files = [Path(p) for p in dark_files]
        self.bad_pix_mask_path = Path(bad_pix_mask_path)
        self.clump_mask_path = Path(clump_mask_path)
        self.coef = coef

    def _check_inputs(self) -> None:
        if not self.dark_files:
            raise InvalidConfigError("No dark files provided for bad pixel mask")
        missing = [str(p) for p in self.dark_files if not p.exists()]
        if missing:
            raise StepExecutionError(f"Missing dark files: {', '.join(missing)}")

    def run(self) -> None:
        """Run the bad pixel and clump mask creation."""
        t0 = time.perf_counter()
        self._check_inputs()
        if not self._quiet:
            self.logger.info("Starting BadPixelMaskCreator")
        self.logger.debug("Dark files: %s", ", ".join(p.name for p in self.dark_files))
        self.logger.debug("Sigma clipping coef: %s", self.coef)

        self.bad_pix_mask_path.parent.mkdir(parents=True, exist_ok=True)
        self.clump_mask_path.parent.mkdir(parents=True, exist_ok=True)

        created: list[str] = []
        if not self._skip_if_exists(self.bad_pix_mask_path):
            self._create_bad_pix_mask()
            created.append(self.bad_pix_mask_path.name)

        if not self._skip_if_exists(self.clump_mask_path):
            self._create_clump_mask()
            created.append(self.clump_mask_path.name)

        if created:
            self.logger.info(
                "Created %s in %s",
                ", ".join(created),
                format_duration(time.perf_counter() - t0),
            )

        if not self._quiet:
            self.logger.info("Finished BadPixelMaskCreator")

    def _read_dark_frames(self) -> np.ndarray:
        frames: list[np.ndarray] = []
        for file_path in self.dark_files:
            with fits.open(file_path) as hdul:
                cube = hdul[0].data
            if cube.ndim == 2:
                if cube.shape[0] == 2048:
                    continue
                if cube.shape[0] == 1024:
                    frames.append(cube)
            elif cube.ndim == 3:
                if cube.shape[1] == 2048:
                    continue
                if cube.shape[1] == 1024:
                    frames.extend(list(cube))
        if not frames:
            raise StepExecutionError("No valid dark frames found for mask creation")
        return np.array(frames)

    def _create_bad_pix_mask(self) -> None:
        dark_frames = self._read_dark_frames()
        sky = np.nanmedian(dark_frames, axis=0)
        med = np.nanmedian(sky)
        sigma = 1.4826 * np.nanmedian(np.abs(sky - med))

        self.logger.info("Bad pixel mask median=%s sigma=%s", med, sigma)

        deadpix = np.where(sky < med - sigma * self.coef)
        hotpix = np.where(sky > med + sigma * self.coef)
        badpix = tuple(np.append(np.array(deadpix), np.array(hotpix), axis=1))

        mask = np.zeros(sky.shape)
        for y, x in zip(badpix[0], badpix[1]):
            mask[y, x] = 1

        hdu = fits.PrimaryHDU(data=mask)
        hdu.writeto(self.bad_pix_mask_path, overwrite=True)

    def _create_clump_mask(self) -> None:
        # Detector geometry and clump positions are instrument-coupled.
        # Pull them from ``config.instrument.detector`` so that a custom
        # instrument profile (different detector shape, different
        # clumps) can be swapped in without editing this method.
        detector = self.config.instrument.detector
        mask_shape = detector.frame_shape
        clump_mask = np.zeros(mask_shape)
        for position in detector.clump_positions:
            clump_mask[
                _circular_aperture_mask(mask_shape, detector.clump_radius, position) == 1
            ] = 1

        hdu = fits.PrimaryHDU(data=clump_mask)
        hdu.writeto(self.clump_mask_path, overwrite=True)
