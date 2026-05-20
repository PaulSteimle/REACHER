"""StarTarget value object.

The legacy code stored per-star, per-night, per-filter state on the
mutable ``Parameters`` god-object (``params.star_name``,
``params.obs_night``, ``params.filter_name``, ``params.fixed_user_mastersky``,
...). That pattern made the call graph hard to follow and prevented the
pipeline from being reasoned about declaratively.

Phase 4 introduces :class:`StarTarget`, a small frozen dataclass that
carries exactly the state a single batch helper needs to process one
``(star_name, obs_night, filter_name)`` triple. The configuration
dataclasses (``ReacherConfig``) remain immutable across the loop;
per-target overrides live on the :class:`StarTarget` instance instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class StarTarget:
    """Immutable per-target value object passed to batch helpers.

    Parameters
    ----------
    star_name
        Science target name. Used to build directory paths and FITS
        provenance keywords.
    obs_night
        Observation night identifier, e.g. ``'2022-10-14'``. Must match
        the directory name on disk.
    filter_name
        IRDIS filter name (the directory name on disk).
    star_center_index
        Index of the CENTER file to use when there is more than one.
        Mirrors the legacy ``params.star_center_index``. Falls back to 0
        if out of bounds.
    fixed_user_mastersky_left
        Optional user-supplied master sky for the LEFT channel. When set
        the master-sky creation step is skipped and the existing file is
        used (mirrors ``params.fixed_user_mastersky['left']``).
    fixed_user_mastersky_right
        Same as above for the RIGHT channel.
    sky_files_exist
        Optional explicit sky-file flag. When ``None`` (default) the
        batch helper auto-detects sky files on disk; when ``True`` /
        ``False`` it overrides that auto-detection (mirrors the legacy
        ``params.sky_files_exist`` flag toggled around the
        ``HD101412`` block).

    Notes
    -----
    The dataclass is ``frozen`` so it can be safely cached / hashed and
    so that batch helpers cannot mutate the call-site target. Any
    mutation that the legacy script did via ``pars.fixed_user_mastersky =
    ...`` becomes a fresh :class:`StarTarget` instance constructed at the
    top of the loop body.
    """

    star_name: str
    obs_night: str
    filter_name: str
    star_center_index: int = 0
    fixed_user_mastersky_left: Optional[str] = None
    fixed_user_mastersky_right: Optional[str] = None
    sky_files_exist: Optional[bool] = None

    def with_overrides(
        self,
        *,
        star_center_index: Optional[int] = None,
        fixed_user_mastersky_left: Optional[str] = None,
        fixed_user_mastersky_right: Optional[str] = None,
        sky_files_exist: Optional[bool] = None,
    ) -> "StarTarget":
        """Return a copy of this target with the given overrides applied."""
        return StarTarget(
            star_name=self.star_name,
            obs_night=self.obs_night,
            filter_name=self.filter_name,
            star_center_index=(
                star_center_index
                if star_center_index is not None
                else self.star_center_index
            ),
            fixed_user_mastersky_left=(
                fixed_user_mastersky_left
                if fixed_user_mastersky_left is not None
                else self.fixed_user_mastersky_left
            ),
            fixed_user_mastersky_right=(
                fixed_user_mastersky_right
                if fixed_user_mastersky_right is not None
                else self.fixed_user_mastersky_right
            ),
            sky_files_exist=(
                sky_files_exist
                if sky_files_exist is not None
                else self.sky_files_exist
            ),
        )

    def fixed_user_mastersky_for(self, channel: str) -> Optional[str]:
        """Return the user-supplied master sky path for ``channel``."""
        if channel == "LEFT":
            return self.fixed_user_mastersky_left
        if channel == "RIGHT":
            return self.fixed_user_mastersky_right
        raise ValueError(f"channel must be 'LEFT' or 'RIGHT'; got {channel!r}")
