"""Per-star calibration-file resolver.

Phase 4 replacement for the legacy
:func:`REACHER.misc.find_calib_files` chain. Per row of the input
DataFrame this function:

1. Tries the exact-night filename (e.g.
   ``bad_pix_mask_<obs_night>.fits``).
2. Falls back to the nearest-in-time calibration file within the
   :attr:`CalibrationConfig.calib_validity_threshold` window.
3. Raises :class:`CalibrationFileNotFoundError` if step 2 fails for a
   required calibration product.

The legacy code duplicated this logic for each of bad-pix mask,
master dark and master flat. Phase 4 dedupes by routing every lookup
through the same :func:`_find_nearest` helper.
"""
from __future__ import annotations

import logging
import os
import re
from glob import glob
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from astropy.time import Time

from ..config import ReacherConfig
from ..exceptions import CalibrationFileNotFoundError, DataFrameError
from .layout import DataLayout


logger = logging.getLogger(__name__)


_STAGE_DICT = {
    "bad_pix": 1,
    "master_dark": 2,
    "master_flat": 3,
    "master_background": 4,
    "science": 5,
    "none": 5,
}


_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _obs_night_jd(obs_night: str) -> float:
    """Convert ``YYYY-MM-DD`` to a Julian Date."""
    return Time(obs_night + "T00:00:00.000", format="isot", scale="utc").jd


def _extract_date_str(filename: str) -> Optional[str]:
    """Pull the first ``YYYY-MM-DD`` substring out of ``filename``.

    The legacy code parsed dates by splitting on ``_`` and indexing
    different positions for each calibration product type
    (``[-1]`` for bad_pix_mask, ``[-2]`` for master_dark,
    ``master_flat_<...>[1][:10]`` for master_flat). That ad-hoc parsing
    is replaced here by a single regex match: every calibration product
    name carries the obs_night as a ``YYYY-MM-DD`` token, so any one of
    them can be located deterministically.
    """
    match = _DATE_RE.search(os.path.basename(filename))
    if match is None:
        return None
    return match.group(0)


def _find_nearest(
    folder: Path,
    pattern: str,
    target_jd: float,
    threshold_days: int,
) -> Optional[str]:
    """Find the calibration file in ``folder`` whose date is closest to ``target_jd``.

    Parameters
    ----------
    folder
        Directory to search.
    pattern
        Glob pattern (e.g. ``master_dark_*_1.0.fits``).
    target_jd
        Target Julian Date.
    threshold_days
        Maximum allowed delta in days. ``None`` is returned if no match
        falls within the window (the caller decides whether that is a
        hard error).

    Returns
    -------
    str or None
        Absolute path to the nearest match, or ``None`` if nothing is
        within the threshold.

    Notes
    -----
    Files whose basename does not contain a ``YYYY-MM-DD`` token are
    skipped (the same forgiving behaviour as the legacy code).
    """
    file_list = sorted(glob(str(folder / pattern)))
    if not file_list:
        return None

    parsed = []
    for path in file_list:
        date_str = _extract_date_str(path)
        if date_str is None:
            continue
        try:
            jd = Time(date_str + "T00:00:00.000", format="isot", scale="utc").jd
        except Exception:
            continue
        parsed.append((path, jd))

    if not parsed:
        return None

    deltas = np.array([abs(jd - target_jd) for _, jd in parsed])
    idx = int(np.argmin(deltas))
    if deltas[idx] > threshold_days:
        return None

    if deltas[idx] > 0:
        logger.warning(
            "using calibration file %s (%.1f days from target)",
            os.path.basename(parsed[idx][0]), deltas[idx],
        )
    return parsed[idx][0]


def _resolve_calib(
    layout: DataLayout,
    config: ReacherConfig,
    *,
    folder: Path,
    exact_filename: str,
    fallback_pattern: str,
    obs_night: str,
    label: str,
    required: bool,
) -> Optional[str]:
    """Resolve a single calibration product for one row.

    Parameters
    ----------
    layout
        Data layout (used to log the folder name only).
    config
        Reacher configuration -- ``calibration.calib_validity_threshold``
        defines the fallback window in days.
    folder
        Calibration product folder (e.g. ``master_flat/``).
    exact_filename
        File name to try on the exact obs_night.
    fallback_pattern
        Glob pattern (``master_flat_*_<filter>.fits`` etc.) used when
        the exact file is missing.
    obs_night
        Target ``YYYY-MM-DD`` night.
    label
        Human-readable label for log messages.
    required
        When ``True`` and the fallback finds nothing within the threshold
        a :class:`CalibrationFileNotFoundError` is raised. When
        ``False`` the function returns ``None`` (this matches the
        legacy 'master darks may be missing for some flats' branch).

    Returns
    -------
    str or None
        Absolute path to the resolved file, or ``None`` when ``required``
        is ``False`` and no calibration is available.
    """
    exact_path = folder / exact_filename
    if exact_path.exists():
        return str(exact_path)

    target_jd = _obs_night_jd(obs_night)
    threshold = config.calibration.calib_validity_threshold
    nearest = _find_nearest(folder, fallback_pattern, target_jd, threshold)

    if nearest is None:
        if required:
            raise CalibrationFileNotFoundError(
                f"{label}: no calibration file within {threshold} days "
                f"for {exact_filename} in {folder}"
            )
        logger.warning(
            "%s: no calibration file within %s days for %s; "
            "leaving the row unset",
            label, threshold, exact_filename,
        )
        return None

    logger.warning(
        "%s: %s not found; using nearest %s instead",
        label, exact_filename, os.path.basename(nearest),
    )
    return nearest


def resolve_star_calibration(
    star_df: pd.DataFrame,
    config: ReacherConfig,
    *,
    layout: Optional[DataLayout] = None,
    calib_type: str = "none",
) -> pd.DataFrame:
    """Augment a per-star DataFrame with calibration product paths.

    Parameters
    ----------
    star_df
        DataFrame produced by :func:`make_star_DF` (or
        :func:`make_flat_DF` when used in the ``master_flat`` stage).
        Must carry ``obs_night``, ``filter`` and ``exp_time`` columns.
    config
        :class:`ReacherConfig`. Reads
        ``calibration.calib_validity_threshold`` for the fallback
        window.
    layout
        Optional :class:`DataLayout` instance; when omitted one is
        constructed from ``config.paths``.
    calib_type
        One of ``'bad_pix' | 'master_dark' | 'master_flat' |
        'master_background' | 'science' | 'none'``. Determines how far
        through the resolution chain we go (mirrors the legacy
        ``stage_dict`` short-circuiting):

        * ``'master_flat'`` (used by Step 1c) resolves bad-pix mask +
          master dark only; rows whose master dark is missing are
          retained with ``master_dark='None'`` so the master-flat batch
          helper can drop them downstream.
        * ``'science'`` / ``'none'`` resolves bad-pix mask + master
          dark + master flat (full chain for Step 2a).

    Returns
    -------
    pandas.DataFrame
        A copy of ``star_df`` with ``bad_pix_mask``, ``master_dark`` (and
        ``master_flat`` for stage >= 4) columns populated with absolute
        paths or ``'None'`` placeholders.

    Raises
    ------
    KeyError
        If ``calib_type`` is unknown.
    CalibrationFileNotFoundError
        If a required calibration product cannot be resolved within the
        validity window.
    DataFrameError
        If the input DataFrame is missing one of the required columns.
    """
    if calib_type not in _STAGE_DICT:
        raise KeyError(
            f"unknown calib_type {calib_type!r}; valid options: "
            f"{sorted(_STAGE_DICT.keys())}"
        )

    required_cols = {"obs_night", "filter", "exp_time"}
    missing_cols = required_cols - set(star_df.columns)
    if missing_cols:
        raise DataFrameError(
            f"resolve_star_calibration: input DataFrame is missing "
            f"columns {missing_cols}"
        )

    if layout is None:
        layout = DataLayout(config.paths)

    stage = _STAGE_DICT[calib_type]
    out = star_df.copy()

    for index, row in out.iterrows():
        obs_night = row["obs_night"]
        filter_name = row["filter"]
        exp_time = row["exp_time"]

        # ---- bad pixel mask (always required) ----
        bad_pix_path = _resolve_calib(
            layout, config,
            folder=layout.bad_pix_mask_folder,
            exact_filename=f"bad_pix_mask_{obs_night}.fits",
            fallback_pattern="bad_pix_mask_*.fits",
            obs_night=obs_night,
            label="bad pixel mask",
            required=True,
        )
        out.loc[index, "bad_pix_mask"] = bad_pix_path

        # ---- master dark ----
        master_dark_required = stage != 3  # master_flat stage tolerates None
        master_dark_path = _resolve_calib(
            layout, config,
            folder=layout.master_dark_folder,
            exact_filename=f"master_dark_{obs_night}_{exp_time}.fits",
            fallback_pattern=f"master_dark_*_{exp_time}.fits",
            obs_night=obs_night,
            label="master dark",
            required=master_dark_required,
        )
        out.loc[index, "master_dark"] = (
            master_dark_path if master_dark_path is not None else "None"
        )

        # ---- master flat (stage >= 4 only) ----
        if stage >= 4:
            master_flat_path = _resolve_calib(
                layout, config,
                folder=layout.master_flat_folder,
                exact_filename=(
                    f"master_flat_{obs_night}_{filter_name}.fits"
                ),
                fallback_pattern=f"master_flat_*_{filter_name}.fits",
                obs_night=obs_night,
                label="master flat",
                required=True,
            )
            out.loc[index, "master_flat"] = master_flat_path

    return out
