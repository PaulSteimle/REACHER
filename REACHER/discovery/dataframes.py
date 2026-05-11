"""Dataset-discovery DataFrame builders.

Phase 4 replacement for the legacy ``REACHER/misc.py`` helpers
(``make_dark_DF`` / ``make_flat_DF`` / ``make_background_DF`` /
``make_star_DF``). The legacy on-disk layout is preserved verbatim::

    raw_input_dir/
        DARK/IRDIS/<obs_night>/<filter>/*.fits
        FLAT/IRDIS/<obs_night>/<filter>/*.fits
        BACKGROUND/IRDIS/<obs_night>/<filter>/*.fits
        <star>/IRDIS/<obs_night>/<filter>/*SPHER*.fits

Each builder returns a :class:`pandas.DataFrame` with one row per FITS
file containing the columns the rest of the pipeline relies on.
"""
from __future__ import annotations

import logging
import os
from glob import glob
from os.path import basename, join
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Optional, Union

import pandas as pd
from astropy.io import fits

from ..config import InstrumentConfig
from ..exceptions import DataFrameError, InvalidConfigError
from .layout import DataLayout, list_directory
from .target import StarTarget


logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Calibration discovery
# ----------------------------------------------------------------------
def _build_calibration_df(
    layout: DataLayout,
    calib_type: str,
    *,
    instrument: InstrumentConfig,
    require_filename_prefix: Optional[str] = None,
    skip_predicate=None,
) -> pd.DataFrame:
    """Walk ``raw_input_dir/<calib_type>/IRDIS/`` and build a DataFrame.

    Parameters
    ----------
    layout
        :class:`DataLayout` carrying the raw input root.
    calib_type
        Folder name under ``raw_input_dir`` (``'DARK'``, ``'FLAT'``,
        ``'BACKGROUND'``).
    require_filename_prefix
        If not ``None``, only files whose basename starts with this
        prefix are included. Used by the BACKGROUND case to match the
        legacy ``BACKGROUND*.fits`` glob.
    skip_predicate
        Optional ``f(header) -> bool`` callback. Files for which it
        returns ``True`` are skipped. Used by ``make_dark_DF`` to drop
        K-band BACKGROUND rows.

    Returns
    -------
    pandas.DataFrame
        Empty when ``raw_input_dir/<calib_type>/IRDIS/`` is missing
        (matches the legacy ``make_background_DF`` behaviour).
    """
    parent_dir = layout.raw_input_dir / calib_type / "IRDIS"
    if not parent_dir.exists():
        logger.warning("calibration dir does not exist: %s", parent_dir)
        return pd.DataFrame()

    rows = []
    for obs_night in list_directory(parent_dir):
        obs_dir = parent_dir / obs_night
        if not obs_dir.is_dir():
            continue
        for filter_name in list_directory(obs_dir):
            filt_dir = obs_dir / filter_name
            if not filt_dir.is_dir():
                continue

            pattern = (
                f"{require_filename_prefix}*.fits"
                if require_filename_prefix
                else "*.fits"
            )
            files = sorted(glob(str(filt_dir / pattern)))
            for file in files:
                filename = basename(file)
                try:
                    header = fits.getheader(file)
                except Exception as exc:
                    raise DataFrameError(
                        f"Could not read header of {file}: {exc}"
                    ) from exc

                if skip_predicate is not None and skip_predicate(header):
                    continue

                row = {
                    "filename": filename,
                    "obs_night": obs_night,
                    "obs_run": obs_night[:-3],
                    "filter": filter_name,
                    "exp_time": header.get("EXPTIME"),
                    "filepath": file,
                    "nd_filter": header.get(instrument.keys.nd_filter_name),
                }
                rows.append(row)

    return pd.DataFrame(rows)


def make_dark_DF(
    raw_input_dir: str | Path,
    *,
    instrument: Optional[InstrumentConfig] = None,
) -> pd.DataFrame:
    """Build the DARK calibration DataFrame.

    Parameters
    ----------
    raw_input_dir
        Root of the raw input layout.
    instrument
        Instrument profile (defaults to SPHERE/IRDIS). Controls the
        FITS header keyword names used to read filter / DPR-type
        metadata and the DPR-type vocabulary for the
        background-substring skip rule.

    Returns
    -------
    pandas.DataFrame
        One row per dark file with columns ``filename``, ``obs_night``,
        ``obs_run``, ``filter``, ``exp_time``, ``filepath``,
        ``nd_filter``.

    Notes
    -----
    K-band ``BACKGROUND`` frames that live under ``DARK/IRDIS/`` are
    excluded -- mirrors the legacy ``make_dark_DF`` skip rule. The
    rule is preserved as a *substring* match (rather than equality
    against a set of DPR types) because the legacy data carries
    composite DPR strings like ``DARK,BACKGROUND``. The substrings to
    match are configured via
    ``instrument.values.background_substrings``.
    """
    if instrument is None:
        instrument = InstrumentConfig.sphere_irdis()
    layout = DataLayout(_PathStub(raw_input_dir))

    background_substrings = instrument.values.background_substrings
    type_key = instrument.keys.dpr_type
    filter_key = instrument.keys.filter_name

    def _skip_k_background(header) -> bool:
        type_str = str(header.get(type_key, ""))
        filter_str = str(header.get(filter_key, ""))
        # Substring match: see docstring.
        type_matches = any(sub in type_str for sub in background_substrings)
        return type_matches and "K" in filter_str

    return _build_calibration_df(
        layout,
        "DARK",
        instrument=instrument,
        skip_predicate=_skip_k_background,
    )


def make_flat_DF(
    raw_input_dir: str | Path,
    *,
    instrument: Optional[InstrumentConfig] = None,
) -> pd.DataFrame:
    """Build the FLAT calibration DataFrame.

    See :func:`make_dark_DF` for the row schema. ``instrument``
    defaults to SPHERE/IRDIS.
    """
    if instrument is None:
        instrument = InstrumentConfig.sphere_irdis()
    layout = DataLayout(_PathStub(raw_input_dir))
    return _build_calibration_df(layout, "FLAT", instrument=instrument)


def make_background_DF(
    raw_input_dir: str | Path,
    *,
    instrument: Optional[InstrumentConfig] = None,
) -> pd.DataFrame:
    """Build the BACKGROUND calibration DataFrame.

    Returns an empty DataFrame when ``raw_input_dir/BACKGROUND/`` does
    not exist (matches the legacy behaviour). ``instrument`` defaults
    to SPHERE/IRDIS.
    """
    if instrument is None:
        instrument = InstrumentConfig.sphere_irdis()
    layout = DataLayout(_PathStub(raw_input_dir))
    return _build_calibration_df(
        layout, "BACKGROUND",
        instrument=instrument,
        require_filename_prefix="BACKGROUND",
    )


# ----------------------------------------------------------------------
# Per-star science discovery
# ----------------------------------------------------------------------
def make_star_DF(
    raw_input_dir: str | Path,
    star_name: str,
    obs_night: str,
    filter_name: str,
    *,
    instrument: Optional[InstrumentConfig] = None,
) -> pd.DataFrame:
    """Build a per-star DataFrame for one ``(obs_night, filter_name)``.

    Parameters
    ----------
    raw_input_dir
        Root of the raw input layout.
    star_name
        Science target name (top-level subdir).
    obs_night
        Observation night identifier (subdir name).
    filter_name
        IRDIS filter name (subdir name).
    instrument
        Instrument profile (defaults to SPHERE/IRDIS). Controls the
        FITS header keyword names used to extract ND filter / DPR-type
        metadata.

    Returns
    -------
    pandas.DataFrame
        One row per ``*SPHER*.fits`` file with columns ``filename``,
        ``obs_night``, ``obs_run``, ``filter``, ``exp_time``,
        ``filepath``, ``nd_filter``, ``type`` (the DPR-type keyword
        configured by ``instrument.keys.dpr_type``).

    Raises
    ------
    DataFrameError
        If any file's header cannot be read.
    """
    if instrument is None:
        instrument = InstrumentConfig.sphere_irdis()

    parent_dir = (
        Path(raw_input_dir)
        / star_name
        / "IRDIS"
        / obs_night
        / filter_name
    )
    rows = []
    files = sorted(glob(str(parent_dir / "*SPHER*.fits")))
    for file in files:
        filename = basename(file)
        try:
            header = fits.getheader(file)
        except Exception as exc:
            raise DataFrameError(
                f"Could not read header of {file}: {exc}"
            ) from exc

        rows.append({
            "filename": filename,
            "obs_night": obs_night,
            "obs_run": obs_night[:-3],
            "filter": filter_name,
            "exp_time": header.get("EXPTIME"),
            "filepath": file,
            "nd_filter": header.get(instrument.keys.nd_filter_name),
            "type": header.get(instrument.keys.dpr_type),
        })

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Star iteration
# ----------------------------------------------------------------------
def list_target_stars(
    raw_input_dir: str | Path,
    *,
    exclude: Optional[Iterable[str]] = None,
) -> list:
    """List science-target subdirectories under ``raw_input_dir``.

    Parameters
    ----------
    raw_input_dir
        Root of the raw input layout.
    exclude
        Iterable of names to exclude. Defaults to the legacy exclusion
        set ``{'FLAT', 'DARK', 'BACKGROUND', 'DB_H23', 'DISTORT',
        'DARK,BACKGROUND'}``.

    Returns
    -------
    list of str
        Sorted list of target star names.
    """
    if exclude is None:
        exclude = {
            "FLAT", "DARK", "BACKGROUND",
            "DB_H23", "DISTORT", "DARK,BACKGROUND",
        }
    else:
        exclude = set(exclude)

    entries = list_directory(raw_input_dir)
    stars = sorted([e for e in entries if e not in exclude])
    return stars


RefCubesValue = Union[None, str, Mapping[str, Optional[str]]]
RefCubesDict = Mapping[str, RefCubesValue]


def stars_from_ref_cubes(ref_cubes: RefCubesDict) -> list[str]:
    """Return the sorted unique set of star names referenced by ``ref_cubes``.

    The ``ref_cubes_dict`` maps each science target to its reference. A
    value may be ``None`` (no reference), a single star name, or a
    mapping ``{obs_night: ref_star_name | None}``. Both keys (science
    targets) and non-``None`` values (references) are returned.

    Parameters
    ----------
    ref_cubes
        Reference-cube mapping of the schema described above.

    Returns
    -------
    list of str
        Sorted unique star names from the keys and the (flattened) values.

    Raises
    ------
    InvalidConfigError
        If a value is neither ``None``, a string, nor a mapping.
    """
    stars: set[str] = set()
    for key, value in ref_cubes.items():
        stars.add(key)
        if value is None:
            continue
        if isinstance(value, str):
            stars.add(value)
        elif isinstance(value, Mapping):
            for v in value.values():
                if v is None:
                    continue
                if not isinstance(v, str):
                    raise InvalidConfigError(
                        f"ref_cubes[{key!r}] has non-str entry: {v!r}"
                    )
                stars.add(v)
        else:
            raise InvalidConfigError(
                f"ref_cubes[{key!r}] must be None, str, or Mapping; "
                f"got {type(value).__name__}"
            )
    return sorted(stars)


def iter_star_targets(
    raw_input_dir: str | Path,
    stars: Union[Iterable[str], RefCubesDict],
    *,
    skip_nd_filters: bool = True,
) -> Iterator[StarTarget]:
    """Yield a :class:`StarTarget` for every ``(star, obs_night, filter)``.

    Parameters
    ----------
    raw_input_dir
        Root of the raw input layout.
    stars
        Either an iterable of star names, or a ``ref_cubes_dict``-style
        mapping (see :func:`stars_from_ref_cubes`). When a mapping is
        passed, both the keys (science targets) and the flattened
        non-``None`` values (reference stars) are iterated.
    skip_nd_filters
        When ``True`` (default), filters whose name contains ``'ND'``
        are skipped -- preserves the legacy
        ``if not 'ND' in filter_name`` rule used in
        ``pre_calibration_and_extraction``.

    Yields
    ------
    StarTarget
        One target per ``(star, obs_night, filter)`` triple.
    """
    raw_input_dir = Path(raw_input_dir)
    if isinstance(stars, Mapping):
        star_names: Iterable[str] = stars_from_ref_cubes(stars)
    else:
        star_names = stars

    for star in star_names:
        star_dir = raw_input_dir / star / "IRDIS"
        if not star_dir.exists():
            logger.warning(
                "star directory does not exist: %s; skipping", star_dir
            )
            continue

        for obs_night in sorted(list_directory(star_dir)):
            obs_dir = star_dir / obs_night
            if not obs_dir.is_dir():
                continue

            for filter_name in sorted(list_directory(obs_dir)):
                filt_dir = obs_dir / filter_name
                if not filt_dir.is_dir():
                    continue
                if skip_nd_filters and "ND" in filter_name:
                    continue
                yield StarTarget(
                    star_name=star,
                    obs_night=obs_night,
                    filter_name=filter_name,
                )


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------
class _PathStub:
    """Adapter used to feed a :class:`DataLayout` from a bare path string."""

    def __init__(self, raw_input_dir: str | Path):
        self.raw_input_dir = str(raw_input_dir)
        self.reduced_output_dir = ""
