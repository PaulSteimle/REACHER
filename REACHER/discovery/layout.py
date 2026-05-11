"""Filesystem layout helpers.

The legacy ``Parameters`` god-object exposed a long list of properties
(``master_dark_folder``, ``stage1_output_folder``,
``ref_left_output_folder``...) that resolved on-disk paths in terms of
the hardcoded layout::

    raw_input_dir/
        DARK/IRDIS/<obs_night>/<filter>/*.fits
        FLAT/IRDIS/<obs_night>/<filter>/*.fits
        BACKGROUND/IRDIS/<obs_night>/<filter>/*.fits
        <star>/IRDIS/<obs_night>/<filter>/*SPHER*.fits

    reduced_output_dir/
        bad_pix_masks/
            bad_pix_mask_<obs_night>.fits
            IRDIS_clump_mask.fits
        master_dark/
            master_dark_<obs_night>_<exp_time>.fits
        master_flat/
            master_flat_<obs_night>_<filter>.fits
        master_background/
        recentering/
        <star>/IRDIS/<obs_night>/<filter>/
            stage1/
            stage2/
            left/raw_files/
            right/raw_files/

This module re-implements that resolver as :class:`DataLayout`, which
takes a :class:`PathConfig` and returns paths via methods (no mutable
state). The legacy property names are preserved verbatim so users
porting old scripts have minimal lift.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..config import PathConfig


def list_directory(path: str | Path):
    """Return entries of a directory ignoring macOS metadata files.

    Mirrors :func:`legacy_misc.list_directory` byte-for-byte: drops any
    entry containing ``.DS`` (catches both ``.DS_Store`` and
    ``._.DS_Store``).

    Parameters
    ----------
    path
        Directory to list.

    Returns
    -------
    list of str
        Entry names (basenames) excluding macOS metadata files.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"directory does not exist: {path}")
    return [f for f in os.listdir(path) if ".DS" not in f]


@dataclass(frozen=True)
class DataLayout:
    """Path resolver mirroring the legacy :class:`Parameters` properties.

    Parameters
    ----------
    paths
        :class:`PathConfig` carrying ``raw_input_dir`` and
        ``reduced_output_dir``.

    Notes
    -----
    All methods return :class:`pathlib.Path` instances; nothing is
    cached; the class is frozen so two callers can construct it from the
    same config without coordinating mutable state.
    """

    paths: PathConfig

    # ------------------------------------------------------------------
    # Top-level reduced output folders
    # ------------------------------------------------------------------
    @property
    def raw_input_dir(self) -> Path:
        return Path(self.paths.raw_input_dir)

    @property
    def reduced_output_dir(self) -> Path:
        return Path(self.paths.reduced_output_dir)

    @property
    def bad_pix_mask_folder(self) -> Path:
        return self.reduced_output_dir / "bad_pix_masks"

    @property
    def master_dark_folder(self) -> Path:
        return self.reduced_output_dir / "master_dark"

    @property
    def master_flat_folder(self) -> Path:
        return self.reduced_output_dir / "master_flat"

    @property
    def master_background_folder(self) -> Path:
        return self.reduced_output_dir / "master_background"

    @property
    def recentering_folder(self) -> Path:
        return self.reduced_output_dir / "recentering"

    @property
    def clump_mask_filepath(self) -> Path:
        return self.bad_pix_mask_folder / "IRDIS_clump_mask.fits"

    # ------------------------------------------------------------------
    # Calibration product names (preserved verbatim from the legacy code)
    # ------------------------------------------------------------------
    def bad_pix_mask_filepath(self, obs_night: str) -> Path:
        return self.bad_pix_mask_folder / f"bad_pix_mask_{obs_night}.fits"

    def master_dark_filepath(self, obs_night: str, exp_time) -> Path:
        return (
            self.master_dark_folder
            / f"master_dark_{obs_night}_{exp_time}.fits"
        )

    def master_flat_filepath(self, obs_night: str, filter_name: str) -> Path:
        return (
            self.master_flat_folder
            / f"master_flat_{obs_night}_{filter_name}.fits"
        )

    def master_background_filepath(
        self,
        obs_night: str,
        filter_name: str,
        exp_time,
        nd_filter: str,
    ) -> Path:
        return (
            self.master_background_folder
            / f"master_background_{obs_night}_{filter_name}_{exp_time}_{nd_filter}.fits"
        )

    # ------------------------------------------------------------------
    # Per-star input layout
    # ------------------------------------------------------------------
    def star_input_folder(
        self, star_name: str, obs_night: str, filter_name: str
    ) -> Path:
        return (
            self.raw_input_dir
            / star_name
            / "IRDIS"
            / obs_night
            / filter_name
        )

    def calib_input_folder(
        self, calib_type: str, obs_night: str, filter_name: str
    ) -> Path:
        """Return the raw input folder for a calibration type.

        ``calib_type`` is one of ``'DARK' | 'FLAT' | 'BACKGROUND'``;
        the legacy layout used these uppercase strings as the folder
        names.
        """
        return (
            self.raw_input_dir
            / calib_type
            / "IRDIS"
            / obs_night
            / filter_name
        )

    # ------------------------------------------------------------------
    # Per-star reduced output layout
    # ------------------------------------------------------------------
    def star_output_folder(
        self, star_name: str, obs_night: str, filter_name: str
    ) -> Path:
        return (
            self.reduced_output_dir
            / star_name
            / "IRDIS"
            / obs_night
            / filter_name
        )

    def stage1_output_folder(
        self, star_name: str, obs_night: str, filter_name: str
    ) -> Path:
        return (
            self.star_output_folder(star_name, obs_night, filter_name)
            / "stage1"
        )

    def stage2_output_folder(
        self, star_name: str, obs_night: str, filter_name: str
    ) -> Path:
        return (
            self.star_output_folder(star_name, obs_night, filter_name)
            / "stage2"
        )

    def left_output_folder(
        self, star_name: str, obs_night: str, filter_name: str
    ) -> Path:
        return (
            self.star_output_folder(star_name, obs_night, filter_name)
            / "left"
            / "raw_files"
        )

    def right_output_folder(
        self, star_name: str, obs_night: str, filter_name: str
    ) -> Path:
        return (
            self.star_output_folder(star_name, obs_night, filter_name)
            / "right"
            / "raw_files"
        )

    def channel_output_folder(
        self,
        star_name: str,
        obs_night: str,
        filter_name: str,
        channel: str,
    ) -> Path:
        if channel == "LEFT":
            return self.left_output_folder(star_name, obs_night, filter_name)
        if channel == "RIGHT":
            return self.right_output_folder(star_name, obs_night, filter_name)
        raise ValueError(f"channel must be 'LEFT' or 'RIGHT'; got {channel!r}")

    # ------------------------------------------------------------------
    # Master sky path resolution (mirrors Parameters.master_sky_filepath)
    # ------------------------------------------------------------------
    def master_sky_filepath(
        self,
        star_name: str,
        obs_night: str,
        filter_name: str,
        channel: str,
        fixed_user_mastersky: Optional[str] = None,
    ) -> Path:
        """Resolve the master-sky path for one channel.

        When ``fixed_user_mastersky`` is supplied (e.g. the user pointed
        at a hand-built master sky) it is returned as-is; otherwise the
        per-target ``stage2/master_sky_<star>_<channel>.fits`` path is
        used (legacy convention).
        """
        if fixed_user_mastersky is not None:
            return Path(fixed_user_mastersky)
        return (
            self.stage2_output_folder(star_name, obs_night, filter_name)
            / f"master_sky_{star_name}_{channel}.fits"
        )
